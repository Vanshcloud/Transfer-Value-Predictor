"""Reusable, table-agnostic validation checks.

Each function takes a frame and returns a list of findings — never raises, never
mutates. Composing them into per-table rule sets lives in
:mod:`src.validation.tables`.
"""

from __future__ import annotations

from collections.abc import Collection, Container, Iterable

import pandas as pd

from src.validation.report import Finding, Severity

# Strings that mean "no value" but are not NaN. This is not hypothetical: the
# `position` column in players.csv uses the literal "Missing" for 37 rows, so a
# null check written with isna() alone passes straight over them.
SENTINEL_STRINGS: frozenset[str] = frozenset(
    {"Missing", "missing", "N/A", "n/a", "NA", "None", "null", "-", ""}
)


def check_required_columns(
    frame: pd.DataFrame, table: str, required: Iterable[str]
) -> list[Finding]:
    """Every column the pipeline depends on must be present."""
    absent = [c for c in required if c not in frame.columns]
    if not absent:
        return []
    return [
        Finding(
            check="required_columns",
            severity=Severity.ERROR,
            table=table,
            message=f"missing required column(s): {', '.join(absent)}",
            count=len(absent),
            unit="columns",
            examples=tuple(absent),
        )
    ]


def check_not_empty(frame: pd.DataFrame, table: str) -> list[Finding]:
    if len(frame):
        return []
    return [Finding("not_empty", Severity.ERROR, table, "table is empty")]


def check_primary_key(frame: pd.DataFrame, table: str, keys: list[str]) -> list[Finding]:
    """The key must be unique.

    A duplicate primary key is always an ERROR: every downstream join silently
    multiplies rows, which inflates a model's training set with copies rather
    than failing.
    """
    present = [k for k in keys if k in frame.columns]
    if len(present) != len(keys):
        return []  # check_required_columns reports the absence

    duplicated = frame.duplicated(subset=keys, keep=False)
    if not duplicated.any():
        return []

    examples = frame.loc[duplicated, keys].head(3).to_dict("records")
    return [
        Finding(
            check="primary_key",
            severity=Severity.ERROR,
            table=table,
            message=f"duplicate primary key ({', '.join(keys)})",
            count=int(duplicated.sum()),
            examples=tuple(examples),
        )
    ]


def check_no_nulls(
    frame: pd.DataFrame,
    table: str,
    columns: Iterable[str],
    *,
    severity: Severity = Severity.ERROR,
) -> list[Finding]:
    """Columns that must always carry a value."""
    findings = []
    for column in columns:
        if column not in frame.columns:
            continue
        nulls = int(frame[column].isna().sum())
        if nulls:
            findings.append(
                Finding(
                    check="no_nulls",
                    severity=severity,
                    table=table,
                    message=f"{column} contains nulls",
                    count=nulls,
                )
            )
    return findings


def check_sentinel_strings(
    frame: pd.DataFrame,
    table: str,
    columns: Iterable[str],
    *,
    sentinels: Collection[str] = SENTINEL_STRINGS,
    severity: Severity = Severity.WARNING,
) -> list[Finding]:
    """Find placeholder strings that a null check would miss.

    The trap this exists for: ``position`` is never NaN, but 37 rows hold the
    literal string ``"Missing"``. Treating that as a real category trains the
    model on a class that means "we do not know".
    """
    findings = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if not (series.dtype == object or isinstance(series.dtype, pd.StringDtype)):
            continue
        hits = series.isin(list(sentinels))
        if hits.any():
            findings.append(
                Finding(
                    check="sentinel_strings",
                    severity=severity,
                    table=table,
                    message=f"{column} uses placeholder strings that are not null",
                    count=int(hits.sum()),
                    examples=tuple(series[hits].unique()[:3]),
                )
            )
    return findings


def check_allowed_values(
    frame: pd.DataFrame,
    table: str,
    column: str,
    allowed: Container[str],
    *,
    severity: Severity = Severity.WARNING,
) -> list[Finding]:
    """Categorical columns must not gain unexpected levels between refreshes."""
    if column not in frame.columns:
        return []
    values = frame[column].dropna()
    unexpected = sorted({v for v in values.unique() if v not in allowed})
    if not unexpected:
        return []
    return [
        Finding(
            check="allowed_values",
            severity=severity,
            table=table,
            message=f"{column} has unexpected value(s)",
            count=int(values.isin(unexpected).sum()),
            examples=tuple(unexpected[:5]),
        )
    ]


def check_range(
    frame: pd.DataFrame,
    table: str,
    column: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    severity: Severity = Severity.WARNING,
) -> list[Finding]:
    """Numeric plausibility. Nulls are ignored; check_no_nulls owns those."""
    if column not in frame.columns:
        return []
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    outside = pd.Series(False, index=series.index)
    if minimum is not None:
        outside |= series < minimum
    if maximum is not None:
        outside |= series > maximum
    if not outside.any():
        return []
    bounds = f"[{minimum if minimum is not None else '-inf'}, {maximum if maximum is not None else 'inf'}]"
    return [
        Finding(
            check="range",
            severity=severity,
            table=table,
            message=f"{column} outside {bounds}",
            count=int(outside.sum()),
            examples=tuple(sorted(series[outside].unique())[:5]),
        )
    ]


def check_parseable_dates(
    frame: pd.DataFrame,
    table: str,
    columns: Iterable[str],
    *,
    severity: Severity = Severity.ERROR,
) -> list[Finding]:
    """Date columns must parse. Values already null are not failures."""
    findings = []
    for column in columns:
        if column not in frame.columns:
            continue
        original_nulls = int(frame[column].isna().sum())
        parsed = pd.to_datetime(frame[column], errors="coerce")
        unparseable = int(parsed.isna().sum()) - original_nulls
        if unparseable > 0:
            bad = frame.loc[parsed.isna() & frame[column].notna(), column]
            findings.append(
                Finding(
                    check="parseable_dates",
                    severity=severity,
                    table=table,
                    message=f"{column} has unparseable dates",
                    count=unparseable,
                    examples=tuple(bad.unique()[:3]),
                )
            )
    return findings
