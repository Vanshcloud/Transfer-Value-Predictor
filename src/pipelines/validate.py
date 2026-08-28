"""Validation as a pipeline stage.

Runs between ingestion and feature engineering. Ingestion deliberately does no
cleaning, so this is the first stage that looks at the data critically — and it
must see it exactly as the provider published it, which is why nothing upstream
coerces dtypes.
"""

from __future__ import annotations

from src.storage.base import TableStore
from src.utils.logging import get_logger
from src.validation.report import ValidationReport
from src.validation.tables import validate_store

logger = get_logger(__name__)


def validate(store: TableStore, *, strict: bool = False) -> ValidationReport:
    """Validate every known table in ``store``.

    Args:
        store: Where the ingested tables live.
        strict: Treat warnings as fatal too. Off by default: the known warnings
            are genuine defects in a third-party dataset that the project
            cannot fix and does not need to — 49 players without a birth date,
            13 with an implausible height, 586 with a placeholder position.
            Failing the pipeline on them would mean it never runs.

    Returns:
        The report. Errors raise; warnings are logged.

    Raises:
        ValidationError: on any ERROR finding, or any finding when ``strict``.
    """
    report = validate_store(store)

    for finding in report.warnings:
        logger.warning("%s", finding.render())

    report.raise_for_errors()

    if strict and report.warnings:
        raise _strict_failure(report)

    logger.info(
        "validation passed: %d error(s), %d warning(s)", len(report.errors), len(report.warnings)
    )
    return report


def _strict_failure(report: ValidationReport) -> Exception:
    from src.validation.report import ValidationError

    detail = "\n".join(f"  {f.render()}" for f in report.warnings)
    return ValidationError(f"strict mode: {len(report.warnings)} warning(s):\n{detail}")
