"""Findings and reports produced by validation.

Validation returns a report rather than raising on the first problem, because a
dataset with four issues should be described in one pass rather than fixed one
exception at a time. The caller decides what is fatal: :meth:`ValidationReport
.raise_for_errors` turns contract violations into an exception, warnings stay
informational.

The distinction is deliberate. A duplicate primary key means every downstream
join is wrong, so it is an error. Thirteen players recorded as 17 cm tall is a
real defect worth surfacing, but it is 0.026% of rows and the model is not
wrong because of it — that is a warning, and preprocessing nulls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    """How much a finding matters."""

    ERROR = "error"
    """A contract violation. Downstream results would be wrong, not merely noisy."""

    WARNING = "warning"
    """A real defect that does not invalidate the data."""

    INFO = "info"
    """Worth recording; no action implied."""


def _plain(value: object) -> object:
    """Unwrap a numpy scalar to its Python equivalent for display.

    Without this, examples render as ``np.float64(17.0)`` — accurate, and noise
    in a report a human is meant to read.
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value


@dataclass(frozen=True)
class Finding:
    """One problem found in one table."""

    check: str
    severity: Severity
    table: str
    message: str
    count: int = 0
    examples: tuple[object, ...] = ()
    unit: str = "rows"
    """What `count` counts. Some findings are about columns, not rows, and
    reporting "4 rows" for four offending columns is quietly wrong."""

    def render(self) -> str:
        location = f"{self.table}" if self.table else "-"
        line = f"[{self.severity.upper():<7}] {location}: {self.message}"
        if self.count:
            line += f" ({self.count:,} {self.unit})"
        if self.examples:
            shown = ", ".join(repr(_plain(e)) for e in self.examples[:5])
            line += f" e.g. {shown}"
        return line


@dataclass
class ValidationReport:
    """Every finding from one validation run."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def of_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def errors(self) -> list[Finding]:
        return self.of_severity(Severity.ERROR)

    @property
    def warnings(self) -> list[Finding]:
        return self.of_severity(Severity.WARNING)

    @property
    def ok(self) -> bool:
        """True when nothing rises to ERROR. Warnings do not make a run fail."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise if any finding is an ERROR.

        Raises:
            ValidationError: listing every error, not just the first.
        """
        if self.errors:
            detail = "\n".join(f"  {f.render()}" for f in self.errors)
            raise ValidationError(f"{len(self.errors)} validation error(s):\n{detail}")

    def render(self) -> str:
        if not self.findings:
            return "validation: no findings"
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        ranked = sorted(self.findings, key=lambda f: (order[f.severity], f.table, f.check))
        summary = f"validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([summary, *(f"  {f.render()}" for f in ranked)])


class ValidationError(RuntimeError):
    """Raised when a dataset violates a contract the pipeline depends on."""
