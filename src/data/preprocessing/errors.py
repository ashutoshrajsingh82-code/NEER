"""
Exception types raised by the NEER preprocessing layer.

Every failure mode of `src/data/preprocessing` surfaces as one of these,
so callers (scripts, training code, the backend API) can catch
`PreprocessingError` once instead of a mixture of `KeyError`,
`ValueError` and `AttributeError`.

`LeakageError` deserves special mention: it is raised when the pipeline
is asked to learn something from data it is not allowed to see. That is
not a user convenience check — it is the guarantee the whole layer is
built around, so it fails loudly rather than warning.
"""

from __future__ import annotations

from typing import Iterable, Optional


class PreprocessingError(Exception):
    """Base class for every error raised by the preprocessing layer."""


class NotFittedError(PreprocessingError):
    """A step that learns from data was used before `fit()` was called.

    Steps that learn nothing (coordinate normalization, masking, feature
    construction) can be used straight away; the ones that do learn
    (missing-value climatologies, normalization statistics) refuse to
    transform anything until they have been fitted, because silently
    computing statistics at transform time is exactly how test data ends
    up influencing training.
    """

    def __init__(self, step: str, what: Optional[str] = None):
        self.step = step
        detail = f" ({what})" if what else ""
        super().__init__(
            f"'{step}' has not been fitted yet{detail}. Call fit() on the training "
            "split first, or use PreprocessingPipeline.run(), which does it for you."
        )


class LeakageError(PreprocessingError):
    """The pipeline was asked to fit on data outside the training split.

    Raised by `splits.assert_fit_window` when the timestamps a step would
    learn from overlap the validation or test period.
    """

    def __init__(self, message: str, offending_times: Optional[Iterable] = None):
        # `offending_times or []` would raise on a NumPy array — the
        # ambiguous-truth-value error — and the guard would fail with a
        # confusing ValueError instead of the LeakageError it meant to
        # raise. The explicit None check is deliberate.
        self.offending_times = [] if offending_times is None else list(offending_times)
        if self.offending_times:
            shown = ", ".join(str(t) for t in self.offending_times[:5])
            more = "" if len(self.offending_times) <= 5 else f" (+{len(self.offending_times) - 5} more)"
            message = f"{message} Offending timestamps: {shown}{more}"
        super().__init__(message)


class StepConfigurationError(PreprocessingError):
    """A step was constructed with an unusable configuration."""
