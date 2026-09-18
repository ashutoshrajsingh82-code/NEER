"""
Exception types raised by the NEER data loading layer.

Every failure mode of `src/data/loaders` surfaces as one of these, so
callers (CLI scripts, the backend API, future training code) can catch
`LoaderError` once instead of catching a mixture of `KeyError`,
`ValueError`, `OSError` and third-party exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data.loaders.validation import ValidationReport


class LoaderError(Exception):
    """Base class for every error raised by the data loading layer."""


class MissingDependencyError(LoaderError):
    """An optional third-party dependency is required but not installed.

    NetCDF support needs `xarray` plus a NetCDF engine (`netCDF4` or
    `h5netcdf`); these are optional so that the CSV / NumPy paths keep
    working in a minimal environment.
    """

    def __init__(self, package: str, purpose: str, install_hint: Optional[str] = None):
        self.package = package
        self.purpose = purpose
        self.install_hint = install_hint or f"pip install {package}"
        super().__init__(
            f"'{package}' is required for {purpose} but is not installed. "
            f"Install it with: {self.install_hint}"
        )


class UnsupportedFormatError(LoaderError):
    """The requested file/extension is not handled by any loader."""


class SchemaError(LoaderError):
    """The source is structurally unusable (missing columns, unnamed dims, ...).

    This is for problems that prevent a dataset from being *built* at all.
    Problems with data that was successfully built (non-monotonic
    coordinates, implausible values, unknown units) are reported through
    `ValidationReport` instead — see `DataValidationError`.
    """


class DataValidationError(LoaderError):
    """A loaded dataset failed validation.

    Carries the full `ValidationReport` so callers can inspect every
    issue, not just the message of the first one.
    """

    def __init__(self, report: "ValidationReport", message: Optional[str] = None):
        self.report = report
        super().__init__(message or report.summary_line())
