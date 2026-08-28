"""Leakage detection — a pipeline stage, not only a test.

Leakage does not raise an exception or produce an obviously wrong number. It
produces a *better* number, which is why it survives review: nobody
investigates a model that beat expectations. These checks run as a stage in the
training pipeline so a leak stops a run instead of quietly improving a metric.

Four failure modes, each observed or specifically anticipated in this dataset:

1. **A feature observed after its label.** Aggregating a whole season and then
   labelling it with a valuation from the middle of that season means the
   features already contain the future.

2. **Current-state columns on historical rows.** ``players.csv`` describes a
   player *now*. Attaching ``contract_expiration_date`` — 37% null and always
   current — to a 2015 row tells the model something nobody knew in 2015. The
   same applies to ``market_value_in_eur`` in that table, which is today's
   value sitting one join away from being today's answer.

3. **The target reaching the feature matrix.** Usually as a rename or a
   transform of itself rather than the raw column.

4. **Overlapping splits.** Both by row and by player: the same player appearing
   in train and test lets the model memorise a person rather than learn a
   pattern.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from src.validation.report import Finding, Severity, ValidationReport

# Columns in players.csv that describe the player as of the last refresh. Safe
# to join for a present-day prediction; never safe on a historical row.
CURRENT_STATE_COLUMNS: frozenset[str] = frozenset(
    {
        "contract_expiration_date",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "current_club_id",
        "current_club_name",
        "current_club_domestic_competition_id",
        "current_national_team_id",
        "last_season",
        "agent_name",
    }
)


def check_feature_time_precedes_label(
    frame: pd.DataFrame,
    *,
    feature_time_column: str,
    label_time_column: str,
    table: str = "training_table",
) -> list[Finding]:
    """Every feature must be observable strictly before its label is set."""
    for column in (feature_time_column, label_time_column):
        if column not in frame.columns:
            return [
                Finding(
                    "leakage_feature_time",
                    Severity.ERROR,
                    table,
                    f"cannot check feature/label ordering: {column} is absent",
                )
            ]

    feature_time = pd.to_datetime(frame[feature_time_column], errors="coerce")
    label_time = pd.to_datetime(frame[label_time_column], errors="coerce")
    comparable = feature_time.notna() & label_time.notna()
    violating = comparable & (feature_time > label_time)

    if not violating.any():
        return []
    return [
        Finding(
            check="leakage_feature_time",
            severity=Severity.ERROR,
            table=table,
            message=(
                f"{feature_time_column} is later than {label_time_column}: "
                "these rows were built from data that did not exist when the label was set"
            ),
            count=int(violating.sum()),
            examples=tuple(frame.loc[violating].index[:3]),
        )
    ]


def check_no_current_state_columns(
    frame: pd.DataFrame,
    *,
    banned: Iterable[str] = CURRENT_STATE_COLUMNS,
    table: str = "training_table",
) -> list[Finding]:
    """Reject columns that describe the present on rows that describe the past."""
    present = sorted(set(frame.columns) & set(banned))
    if not present:
        return []
    return [
        Finding(
            check="leakage_current_state",
            severity=Severity.ERROR,
            table=table,
            message=(
                "current-state column(s) present on historical rows: "
                f"{', '.join(present)} — these describe the player now, not during the season"
            ),
            count=len(present),
            unit="columns",
            examples=tuple(present),
        )
    ]


def check_target_absent_from_features(
    feature_columns: Sequence[str],
    target_column: str,
    *,
    table: str = "training_table",
) -> list[Finding]:
    """The target, or an obvious transform of it, must not be a feature.

    Matches on the target's stem, so ``log_market_value_in_eur`` and
    ``market_value_in_eur_scaled`` are caught alongside the raw column. A
    deliberately lagged feature must therefore be named to say so — ``prev_``
    and ``lag_`` prefixes are allowed.
    """
    allowed_prefixes = ("prev_", "lag_", "prior_")
    stem = target_column.removeprefix("log_").removeprefix("log1p_")

    offenders = [
        column
        for column in feature_columns
        if stem in column and not column.startswith(allowed_prefixes)
    ]
    if not offenders:
        return []
    return [
        Finding(
            check="leakage_target_in_features",
            severity=Severity.ERROR,
            table=table,
            message=(
                f"target {target_column!r} reaches the feature matrix as: {', '.join(offenders)}. "
                f"A lagged feature must be named with one of {allowed_prefixes}"
            ),
            count=len(offenders),
            unit="columns",
            examples=tuple(offenders),
        )
    ]


def check_splits_are_disjoint(
    splits: dict[str, pd.Index],
    *,
    groups: pd.Series | None = None,
    table: str = "splits",
) -> list[Finding]:
    """Splits must not share rows, and optionally must not share players."""
    findings: list[Finding] = []
    names = sorted(splits)

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = splits[left].intersection(splits[right])
            if len(shared):
                findings.append(
                    Finding(
                        check="leakage_split_overlap",
                        severity=Severity.ERROR,
                        table=table,
                        message=f"{left} and {right} share rows",
                        count=len(shared),
                        examples=tuple(shared[:3]),
                    )
                )

    if groups is not None:
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared_groups = set(groups.loc[splits[left]]) & set(groups.loc[splits[right]])
                if shared_groups:
                    findings.append(
                        Finding(
                            check="leakage_group_overlap",
                            severity=Severity.WARNING,
                            table=table,
                            message=(
                                f"{left} and {right} share {len(shared_groups)} player(s); "
                                "expected under a temporal split, where a career spans the boundary"
                            ),
                            count=len(shared_groups),
                            unit="players",
                            examples=tuple(sorted(shared_groups)[:3]),
                        )
                    )

    return findings


def detect_leakage(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    feature_time_column: str | None = None,
    label_time_column: str | None = None,
    splits: dict[str, pd.Index] | None = None,
    groups: pd.Series | None = None,
) -> ValidationReport:
    """Run every leakage check that the supplied arguments make possible."""
    report = ValidationReport()
    report.extend(check_no_current_state_columns(frame[list(feature_columns)]))
    report.extend(check_target_absent_from_features(feature_columns, target_column))

    if feature_time_column and label_time_column:
        report.extend(
            check_feature_time_precedes_label(
                frame,
                feature_time_column=feature_time_column,
                label_time_column=label_time_column,
            )
        )
    if splits:
        report.extend(check_splits_are_disjoint(splits, groups=groups))
    return report
