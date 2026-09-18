"""
The contract every NEER preprocessing step implements.

A step is a reusable object that takes an `OceanDataset` and returns a
new `OceanDataset`. Steps compose into a `PreprocessingPipeline`
(`pipeline.py`), and each one answers three questions about itself:

* **What was I configured to do?** — `config()`
* **What did I learn from the data?** — `state()`
* **What did I actually do last time?** — `report()`

Those three feed straight into the saved preprocessing metadata, so a
tensor file on disk can always be traced back to the exact settings and
the exact fitted statistics that produced it.

fit / transform, and why the split matters
------------------------------------------
The interface is deliberately the familiar scikit-learn one:

    step.fit(train_dataset)          # learn parameters, training data only
    step.transform(any_dataset)      # apply them

`learns_from_data` is the flag that makes leakage checkable rather than a
matter of trust. A step that sets it False is a pure function of its
input — coordinate normalization, temporal alignment, masking, feature
construction — and can be applied to any data at any time. A step that
sets it True (`MissingValueHandler`, `Normalizer`) holds parameters
derived from data, must be fitted before it will transform anything, and
is only ever fitted on the training split by `PreprocessingPipeline`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional

from src.data.loaders.representation import OceanDataset
from src.data.preprocessing._utils import json_safe
from src.data.preprocessing.errors import NotFittedError


class PreprocessingStep(ABC):
    """One reusable stage of the NEER preprocessing pipeline."""

    #: Short machine name, used as the key in saved metadata.
    name: ClassVar[str] = "step"

    #: Human-readable title for reports.
    title: ClassVar[str] = "Preprocessing step"

    #: True when this step derives parameters from data it is shown, and
    #: therefore must only ever be fitted on the training split.
    learns_from_data: ClassVar[bool] = False

    def __init__(self) -> None:
        self._fitted = False
        self._report: Dict[str, Any] = {}
        self._fit_summary: Dict[str, Any] = {}

    # -- fit / transform ---------------------------------------------------

    @property
    def fitted(self) -> bool:
        """Whether `fit` has been called (always True for stateless steps)."""
        return self._fitted or not self.learns_from_data

    def fit(self, dataset: OceanDataset) -> "PreprocessingStep":
        """Learn this step's parameters from `dataset`.

        `dataset` must be the *training* split — the pipeline enforces
        that. Stateless steps accept the call and record only the shape
        of what they saw.
        """
        self._fit_summary = {
            "n_time": dataset.n_time,
            "sizes": dict(dataset.sizes),
            "variables": dataset.variable_names,
        }
        self._fit(dataset)
        self._fitted = True
        return self

    def _fit(self, dataset: OceanDataset) -> None:
        """Hook for subclasses that learn something. Default: nothing to learn."""
        return None

    def transform(self, dataset: OceanDataset) -> OceanDataset:
        """Apply this step, returning a new dataset. Never mutates the input."""
        if self.learns_from_data and not self._fitted:
            raise NotFittedError(self.name)
        return self._transform(dataset)

    @abstractmethod
    def _transform(self, dataset: OceanDataset) -> OceanDataset:
        """Subclass implementation of `transform`."""

    def fit_transform(self, dataset: OceanDataset) -> OceanDataset:
        """`fit` then `transform` on the same (training) dataset."""
        return self.fit(dataset).transform(dataset)

    # -- introspection -----------------------------------------------------

    def config(self) -> Dict[str, Any]:
        """The step's configuration, as JSON-safe values."""
        return {}

    def state(self) -> Dict[str, Any]:
        """What the step learned, summarized for metadata.

        Summarized, not dumped: a climatology field is megabytes of
        floats and belongs in the tensor file, not in a metadata JSON.
        Enough is recorded to tell two fitted pipelines apart.
        """
        return {}

    def report(self) -> Dict[str, Any]:
        """What the last `transform` actually did (counts, coverage, ...)."""
        return dict(self._report)

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-safe description: config, learned state, last report."""
        return json_safe(
            {
                "step": self.name,
                "title": self.title,
                "class": type(self).__name__,
                "learns_from_data": self.learns_from_data,
                "fitted": self.fitted,
                "fit_summary": self._fit_summary,
                "config": self.config(),
                "state": self.state(),
                "report": self._report,
            }
        )

    def describe(self) -> str:
        """One-line human-readable description, for logs."""
        marker = "learned" if self.learns_from_data else "stateless"
        settings = ", ".join(f"{k}={v}" for k, v in self.config().items())
        return f"{self.name} [{marker}] {settings}".strip()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} fitted={self.fitted}>"


class StatelessStep(PreprocessingStep):
    """Convenience base for steps that are a pure function of their input.

    Subclasses only implement `_transform` (and usually `config`). They
    can be applied to the full dataset before the split is even decided,
    because there is nothing for them to learn and therefore nothing to
    leak.
    """

    learns_from_data: ClassVar[bool] = False


class LearnedStep(PreprocessingStep):
    """Convenience base for steps that hold parameters derived from data.

    Subclasses implement `_fit` and `_transform`, and must be fitted on
    the training split before use — `transform` raises `NotFittedError`
    otherwise.
    """

    learns_from_data: ClassVar[bool] = True

    def require_fitted(self, what: Optional[str] = None) -> None:
        if not self._fitted:
            raise NotFittedError(self.name, what)
