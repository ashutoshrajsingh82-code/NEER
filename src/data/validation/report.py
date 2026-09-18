"""
The validation report: statuses, findings, and the report object itself.

Phase 06 requires a three-level outcome — **valid / warning / error** — so
that is the primary vocabulary here, at two levels of granularity:

* every rule produces one `CheckResult` with its own status, and
* the report's overall status is the worst status among its checks.

Severity policy
---------------
The split is the whole point of the layer, so it is stated once here and
applied consistently by every rule in `rules.py`:

* **ERROR** — the data is scientifically wrong or unusable as NEER input.
  Latitudes outside ±90°, a non-monotonic axis, duplicate timestamps,
  duplicate coordinates, a required variable missing, a variable that is
  entirely NaN, a unit that contradicts the variable it labels.
* **WARNING** — the data is usable but something needs a human decision.
  Coordinates outside the configured NEER domain, a grid resolution that
  differs from the configured one, an irregular time cadence, depth levels
  that do not match the configured set, undeclared units, high (but not
  total) NaN fractions, values outside a plausible physical range.
* **VALID** — the rule found nothing to report.

No repairs
----------
A `Finding` may carry a `remedy`: a suggestion in prose, for a human to
act on. Nothing in this package ever applies one. Validation takes a
dataset and returns a report; the dataset it was given is never modified.
`rules.py` has no write path, and
`tests/test_validation_demo.py::test_validation_never_modifies_the_dataset`
holds that line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


class ValidationStatus(str, Enum):
    """The three outcome levels required by Phase 06."""

    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return {"valid": 0, "warning": 1, "error": 2}[self.value]

    @property
    def symbol(self) -> str:
        return {"valid": "✓", "warning": "⚠", "error": "✗"}[self.value]

    @classmethod
    def worst(cls, statuses: Iterable["ValidationStatus"]) -> "ValidationStatus":
        """The most severe status in `statuses` (VALID when empty)."""
        return max(statuses, key=lambda s: s.rank, default=cls.VALID)


@dataclass(frozen=True)
class Finding:
    """One thing a rule found.

    `remedy` is advisory only — a suggestion for a human. Nothing in this
    package acts on it.
    """

    status: ValidationStatus
    message: str
    target: Optional[str] = None  # the variable or coordinate concerned
    remedy: Optional[str] = None

    def __str__(self) -> str:
        where = f" [{self.target}]" if self.target else ""
        return f"{self.status.symbol} {self.status.value.upper()}{where}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "target": self.target,
            "message": self.message,
            "remedy": self.remedy,
        }


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one validation rule.

    A rule that found nothing still produces a result — with status VALID
    — so the report shows what was actually checked rather than only what
    failed. `details` holds the numbers the rule measured (observed
    resolution, NaN fractions, ...) for the report and for tests.
    """

    name: str
    title: str
    findings: List[Finding] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    skipped_reason: Optional[str] = None

    @property
    def status(self) -> ValidationStatus:
        if self.skipped_reason is not None:
            return ValidationStatus.VALID
        return ValidationStatus.worst(f.status for f in self.findings)

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None

    def of(self, status: ValidationStatus) -> List[Finding]:
        return [f for f in self.findings if f.status is status]

    @property
    def errors(self) -> List[Finding]:
        return self.of(ValidationStatus.ERROR)

    @property
    def warnings(self) -> List[Finding]:
        return self.of(ValidationStatus.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.name,
            "title": self.title,
            "status": self.status.value,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
            "details": self.details,
            "findings": [f.to_dict() for f in self.findings],
        }


class ValidationError(Exception):
    """Raised by `ValidationReport.raise_for_status` when validation fails.

    Carries the report, so a caller that catches it can inspect every
    finding rather than just the message of the first.
    """

    def __init__(self, report: "ValidationReport", message: Optional[str] = None):
        self.report = report
        super().__init__(message or report.summary_line())


@dataclass
class ValidationReport:
    """The full outcome of validating one dataset.

    Renderable as a dict (`to_dict`), JSON, plain text (`to_text`) or
    Markdown (`to_markdown`), and writable to disk with `save` — the
    report is meant to be an artifact that can be attached to a data
    ingest, not only an in-process object.
    """

    checks: List[CheckResult] = field(default_factory=list)
    dataset_source: Optional[str] = None
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # -- building ----------------------------------------------------------

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    # -- status ------------------------------------------------------------

    @property
    def status(self) -> ValidationStatus:
        """The worst status across every check."""
        return ValidationStatus.worst(c.status for c in self.checks)

    @property
    def is_valid(self) -> bool:
        """True only when nothing at all was reported (no warnings either)."""
        return self.status is ValidationStatus.VALID

    @property
    def has_errors(self) -> bool:
        return self.status is ValidationStatus.ERROR

    @property
    def is_usable(self) -> bool:
        """True when there are no errors; warnings are tolerated."""
        return not self.has_errors

    # -- querying ----------------------------------------------------------

    def findings(self, status: Optional[ValidationStatus] = None) -> List[Finding]:
        out = [f for c in self.checks for f in c.findings]
        if status is not None:
            out = [f for f in out if f.status is status]
        return out

    @property
    def errors(self) -> List[Finding]:
        return self.findings(ValidationStatus.ERROR)

    @property
    def warnings(self) -> List[Finding]:
        return self.findings(ValidationStatus.WARNING)

    def check(self, name: str) -> CheckResult:
        """The result of one named check."""
        for result in self.checks:
            if result.name == name:
                return result
        raise KeyError(
            f"no check named '{name}' in this report; ran: {[c.name for c in self.checks]}"
        )

    def __contains__(self, name: object) -> bool:
        return any(c.name == name for c in self.checks)

    def checks_by_status(self, status: ValidationStatus) -> List[CheckResult]:
        return [c for c in self.checks if c.status is status]

    def remedies(self) -> List[str]:
        """Suggested fixes for everything that was reported.

        Advisory output for a human; this layer never applies them.
        """
        seen: List[str] = []
        for finding in self.findings():
            if finding.remedy and finding.remedy not in seen:
                seen.append(finding.remedy)
        return seen

    def counts(self) -> Dict[str, int]:
        return {
            "checks": len(self.checks),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    def summary_line(self) -> str:
        counts = self.counts()
        return (
            f"{self.status.value.upper()} — {counts['checks']} check(s), "
            f"{counts['errors']} error(s), {counts['warnings']} warning(s)"
        )

    def raise_for_status(self, *, strict: bool = False) -> None:
        """Raise `ValidationError` if the report has errors.

        With `strict=True`, warnings count as failures too.
        """
        failures = list(self.errors)
        if strict:
            failures += self.warnings
        if not failures:
            return
        source = f" for {self.dataset_source}" if self.dataset_source else ""
        detail = "\n".join(f"  - {f}" for f in failures)
        raise ValidationError(
            self, f"Data validation failed{source} ({self.summary_line()}):\n{detail}"
        )

    # -- rendering ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "is_usable": self.is_usable,
            "source": self.dataset_source,
            "created_at": self.created_at,
            "counts": self.counts(),
            "dataset": self.dataset_summary,
            "checks": [c.to_dict() for c in self.checks],
            "remedies": self.remedies(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        lines = [
            "NEER data validation report",
            "=" * 60,
            f"source : {self.dataset_source or '-'}",
            f"time   : {self.created_at}",
            f"status : {self.status.symbol} {self.status.value.upper()}",
            f"summary: {self.summary_line()}",
            "",
        ]
        for result in self.checks:
            suffix = f" (skipped: {result.skipped_reason})" if result.skipped else ""
            lines.append(f"{result.status.symbol} {result.title}{suffix}")
            for finding in result.findings:
                target = f"[{finding.target}] " if finding.target else ""
                lines.append(f"    {finding.status.value:<7} {target}{finding.message}")
                if finding.remedy:
                    lines.append(f"            → {finding.remedy}")
        if self.is_valid:
            lines += ["", "No problems found."]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        counts = self.counts()
        lines = [
            "# NEER data validation report",
            "",
            f"- **Status:** {self.status.symbol} `{self.status.value.upper()}`",
            f"- **Source:** `{self.dataset_source or '-'}`",
            f"- **Generated:** {self.created_at}",
            f"- **Checks:** {counts['checks']} · "
            f"**Errors:** {counts['errors']} · **Warnings:** {counts['warnings']}",
            "",
            "| Check | Status | Findings |",
            "| --- | --- | --- |",
        ]
        for result in self.checks:
            lines.append(
                f"| {result.title} | {result.status.symbol} "
                f"{result.status.value} | {len(result.findings)} |"
            )

        reported = [c for c in self.checks if c.findings]
        if reported:
            lines += ["", "## Details", ""]
            for result in reported:
                lines += [f"### {result.title}", ""]
                for finding in result.findings:
                    target = f"`{finding.target}` — " if finding.target else ""
                    lines.append(
                        f"- **{finding.status.value.upper()}** {target}{finding.message}"
                    )
                    if finding.remedy:
                        lines.append(f"  - *Suggested fix:* {finding.remedy}")
                lines.append("")
        else:
            lines += ["", "No problems found.", ""]

        if self.remedies():
            lines += ["## Suggested fixes", ""]
            lines += [f"{i}. {r}" for i, r in enumerate(self.remedies(), start=1)]
            lines += [
                "",
                "> These are suggestions only. This layer never modifies data.",
                "",
            ]
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the report to disk, choosing the format from the extension.

        `.json` → JSON, `.md` → Markdown, anything else → plain text.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".json":
            text = self.to_json()
        elif suffix in (".md", ".markdown"):
            text = self.to_markdown()
        else:
            text = self.to_text()
        path.write_text(text, encoding="utf-8")
        return path

    def __str__(self) -> str:
        return self.to_text()

    def __len__(self) -> int:
        return len(self.checks)


def merge_statuses(statuses: Sequence[ValidationStatus]) -> ValidationStatus:
    """Convenience wrapper around `ValidationStatus.worst`."""
    return ValidationStatus.worst(statuses)
