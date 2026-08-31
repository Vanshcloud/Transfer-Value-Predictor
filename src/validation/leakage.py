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

5. **Duplicate entity rows.** Two rows for the same player-season are the same
   observation twice. Any split that is not grouped then puts one copy in train
   and the other in test, and the model is scored on rows it has already seen.

6. **A lagged feature that is not actually lagged.** ``prev_market_value_in_eur``
   is a deliberate copy of the target from an earlier season, and the name is
   what the target check accepts as proof of that. A name is a promise; the
   dates are the audit. Widening the label window to a year made it possible
   for the previous season's label to be published after the current season's
   features closed.

7. **A date column holding a future date.** The current-state check works on
   column *names*, so it catches ``contract_expiration_date`` and misses the
   same column joined in under another name, or a transfer date arriving with
   Phase 12's enrichment. This one works on *values*: any date in the feature
   matrix that falls after the label date describes something that had not
   happened yet.

:class:`LeakageValidator` bundles all seven behind one configured object, so a
pipeline stage states its column contract once and every later stage re-runs
the identical checks rather than a hand-copied subset of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

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


def check_no_duplicate_entities(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    table: str = "training_table",
) -> list[Finding]:
    """One observation per entity. Two rows for one player-season are one row twice.

    This is leakage rather than untidiness: a random or grouped split will put
    one copy in train and the other in test, and the model is then scored on a
    row it has already been fitted to.
    """
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        return [
            Finding(
                check="leakage_duplicate_entity",
                severity=Severity.ERROR,
                table=table,
                message=f"cannot check for duplicate rows: {', '.join(missing)} absent",
                count=len(missing),
                unit="columns",
            )
        ]

    duplicated = frame.duplicated(subset=list(keys), keep=False)
    if not duplicated.any():
        return []

    offenders = frame.loc[duplicated, list(keys)]
    return [
        Finding(
            check="leakage_duplicate_entity",
            severity=Severity.ERROR,
            table=table,
            message=(
                f"duplicate rows for the same ({', '.join(keys)}): the same observation "
                "appears more than once and will straddle any split"
            ),
            count=int(duplicated.sum()),
            examples=tuple(offenders.head(3).itertuples(index=False, name=None)),
        )
    ]


def check_no_future_dates(
    frame: pd.DataFrame,
    *,
    label_time_column: str,
    columns: Iterable[str] | None = None,
    table: str = "training_table",
) -> list[Finding]:
    """No date *value* in the feature matrix may postdate the label.

    The complement of :func:`check_no_current_state_columns`, which works on
    column names. This one works on values, so it still fires when a contract
    expiry or a transfer date arrives under a name nobody thought to ban —
    which is the realistic way such a column gets in.

    Run over the *whole* frame by :class:`LeakageValidator`, not the feature
    subset. That is what lets it audit a numeric feature indirectly: an as-of
    join that leaves its source date behind as a column becomes checkable by
    value, and a join whose direction silently regresses fails here rather than
    in a metric six weeks later.
    """
    if label_time_column not in frame.columns:
        return [
            Finding(
                check="leakage_future_date",
                severity=Severity.ERROR,
                table=table,
                message=f"cannot check date values: {label_time_column} is absent",
            )
        ]

    label_time = pd.to_datetime(frame[label_time_column], errors="coerce")
    candidates = frame.columns if columns is None else [c for c in columns if c in frame.columns]

    findings: list[Finding] = []
    for column in candidates:
        if column == label_time_column or not pd.api.types.is_datetime64_any_dtype(frame[column]):
            continue

        values = frame[column]
        violating = values.notna() & label_time.notna() & (values > label_time)
        if not violating.any():
            continue

        findings.append(
            Finding(
                check="leakage_future_date",
                severity=Severity.ERROR,
                table=table,
                message=(
                    f"{column} holds dates later than {label_time_column}: "
                    "this column describes something that had not happened when the label was set"
                ),
                count=int(violating.sum()),
                examples=tuple(values.loc[violating].head(3)),
            )
        )
    return findings


def check_lagged_values_precede_features(
    frame: pd.DataFrame,
    *,
    lag_age_column: str = "prev_value_age_days",
    table: str = "training_table",
) -> list[Finding]:
    """A lagged feature must describe something already published.

    The ``prev_`` prefix is what
    :func:`check_target_absent_from_features` accepts as proof that a copy of
    the target is deliberately lagged. That check reads the *name*. This one
    reads the *dates*, because a name is a promise and this is the audit.

    Phase 15 widened the label window from 120 days to a year, and a wider
    window means season *s* can be labelled after season *s+1* has already
    started — at which point *s+1*'s "previous" value is not previous. It
    affected 22 rows in 61,555, which is precisely the size of defect that
    survives review: too small to show in a metric, large enough to be wrong.

    A non-positive staleness is the signature: the prior value was published on
    or after the moment the features closed.
    """
    if lag_age_column not in frame.columns:
        return []

    ages = pd.to_numeric(frame[lag_age_column], errors="coerce")
    violating = ages.notna() & (ages <= 0)
    if not violating.any():
        return []
    return [
        Finding(
            check="leakage_lagged_value_not_lagged",
            severity=Severity.ERROR,
            table=table,
            message=(
                f"{lag_age_column} is not positive: the lagged value was published on or "
                "after the moment the features were observed, so it is present information"
            ),
            count=int(violating.sum()),
            examples=tuple(frame.loc[violating].index[:3]),
        )
    ]


@dataclass(frozen=True)
class LeakageValidator:
    """Every leakage check this project knows about, behind one configured object.

    A stage declares its column contract once — what the features are, what the
    target is, which timestamps bound them, what makes a row unique — and every
    later stage re-runs exactly those checks by reusing the validator, instead
    of re-passing eight keyword arguments and quietly omitting one.

    Checks whose inputs are absent are skipped rather than failed: a frame with
    no timestamps cannot be ordered, and reporting that as a leak would train
    people to ignore the report. What is *never* skipped is a check whose inputs
    were promised and then turned out to be missing — that is an error, because
    a silently absent contract is how a check stops running without anyone
    noticing.
    """

    feature_columns: Sequence[str]
    target_column: str
    feature_time_column: str | None = None
    label_time_column: str | None = None
    entity_keys: Sequence[str] = ()
    banned_columns: frozenset[str] = CURRENT_STATE_COLUMNS
    table: str = "training_table"

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        splits: dict[str, pd.Index] | None = None,
        groups: pd.Series | None = None,
    ) -> ValidationReport:
        """Run every check the supplied frame and configuration make possible."""
        report = ValidationReport()
        present = [column for column in self.feature_columns if column in frame.columns]

        missing = sorted(set(self.feature_columns) - set(present))
        if missing:
            report.add(
                Finding(
                    check="leakage_missing_features",
                    severity=Severity.ERROR,
                    table=self.table,
                    message=(
                        f"declared feature column(s) absent: {', '.join(missing)} — "
                        "the checks that would have covered them did not run"
                    ),
                    count=len(missing),
                    unit="columns",
                    examples=tuple(missing),
                )
            )

        features = frame.loc[:, present]
        report.extend(
            check_no_current_state_columns(features, banned=self.banned_columns, table=self.table)
        )
        report.extend(
            check_target_absent_from_features(present, self.target_column, table=self.table)
        )

        if self.entity_keys:
            report.extend(check_no_duplicate_entities(frame, self.entity_keys, table=self.table))

        if self.feature_time_column and self.label_time_column:
            report.extend(
                check_feature_time_precedes_label(
                    frame,
                    feature_time_column=self.feature_time_column,
                    label_time_column=self.label_time_column,
                    table=self.table,
                )
            )
        report.extend(check_lagged_values_precede_features(frame, table=self.table))

        if self.label_time_column:
            # The whole frame, not just the feature columns. A feature can be a
            # numeric aggregate whose *inputs* postdate the label — a club's
            # points per game averaged over a season that runs a fortnight past
            # the row's as-of date — and no check on the feature's own name or
            # dtype can see that. What can see it is the provenance timestamp
            # the join leaves behind (`club_record_date`), which is a date value
            # in the frame and is not a feature. Checking only the features
            # skipped exactly the column that would have caught it: measured on
            # the full panel, the whole-season club aggregate folded post-label
            # matches into 1,049 rows for a year before anyone looked.
            report.extend(
                check_no_future_dates(
                    frame,
                    label_time_column=self.label_time_column,
                    columns=None,
                    table=self.table,
                )
            )
        if splits:
            report.extend(check_splits_are_disjoint(splits, groups=groups))

        return report

    def raise_for_leakage(
        self,
        frame: pd.DataFrame,
        *,
        splits: dict[str, pd.Index] | None = None,
        groups: pd.Series | None = None,
    ) -> ValidationReport:
        """Validate and raise on any error. Returns the report when it is clean.

        Raises:
            ValidationError: listing every leak found, not just the first.
        """
        report = self.validate(frame, splits=splits, groups=groups)
        report.raise_for_errors()
        return report


def detect_leakage(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    feature_time_column: str | None = None,
    label_time_column: str | None = None,
    entity_keys: Sequence[str] = (),
    splits: dict[str, pd.Index] | None = None,
    groups: pd.Series | None = None,
) -> ValidationReport:
    """One-shot :class:`LeakageValidator` for callers with nothing to reuse."""
    validator = LeakageValidator(
        feature_columns=feature_columns,
        target_column=target_column,
        feature_time_column=feature_time_column,
        label_time_column=label_time_column,
        entity_keys=entity_keys,
    )
    return validator.validate(frame, splits=splits, groups=groups)
