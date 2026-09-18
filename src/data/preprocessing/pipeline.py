"""
The NEER preprocessing pipeline: ordering, fitting, and the leakage rule.

This module is where the individual steps become a process with a
guarantee. Each step knows how to do its own job; the pipeline decides
what order they run in, which data each one is allowed to learn from, and
what gets written down about the run.

The order, and why it is this order
-----------------------------------
::

    raw dataset
      1. coordinate normalization   stateless
      2. temporal alignment         stateless
      -- the split is decided here --
      3. spatial alignment          stateless
      4. missing-value handling     LEARNED  (climatology)
      5. masking                    stateless
      6. feature construction       stateless
      7. normalization              LEARNED  (per-variable statistics)
      8. tensor assembly
    normalized tensors + metadata

Coordinates come first because every later stage assumes ascending axes
and one longitude convention. Temporal alignment comes second because the
split is defined by timestamps, and timestamps only have stable identities
once everything is on a common cadence — deciding the split first and
then resampling the time axis underneath it is how a boundary quietly
moves. Missing-value handling precedes masking so that masking can record
what was genuinely observed before the gaps were filled, and feature
construction follows both so that features are built from complete
fields rather than propagating holes into every derived channel.
Normalization is last because it should see the features too.

The leakage rule
----------------
Two sentences:

1. **A step that learns is fitted on the training period and nothing
   else.**
2. **After the split, every period is transformed on its own.**

The first is the obvious one, and the pipeline enforces it mechanically
rather than by convention: before any `fit`, it slices the dataset down
to the training timesteps and then calls `assert_fit_window` on the
timestamps *in that slice* — not on the timestamps it meant to pass. If
anything outside the training period is in there, `LeakageError` is
raised and the run stops.

The second is the one that is easy to miss, and it is why the post-split
stages run three times instead of once. Fitting on training data is not
sufficient on its own, because a *stateless* step can leak too if it
mixes timesteps: `MissingValueHandler` bridges short gaps by
interpolating along the time axis, and a hole in the last training month
sitting between two observed months would be filled from the first
validation month. No parameter was fitted, no guard was tripped, and
validation data is now inside the training tensor.

Transforming each period separately removes that whole class of bug by
construction rather than by inspection. Train, validation and test never
appear in the same array after the split, so no step — present or future,
careful or not — can reach across a boundary. The periods are
concatenated back together only at the very end, for tensor assembly,
where the operation is a repacking and the split travels along as masks.

The price is a small, correct loss of fill quality: a gap at the start of
the test period can no longer be bridged from the last training month and
falls through to the climatology instead. That is the right trade — the
alternative is a test score that is partly a memory of training data.

What the pipeline does *not* do
-------------------------------
It does not shuffle, it does not offer a random split, and it does not
fit anything on the full dataset "just for normalization". Those are the
three ways this kind of code usually leaks, and none of them is reachable
through this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.data.loaders.representation import OceanDataset
from src.data.preprocessing._utils import (
    axis_of,
    json_safe,
    select_times,
    with_values,
    with_variables,
)
from src.data.preprocessing.base import PreprocessingStep
from src.data.preprocessing.coordinates import CoordinateNormalizer
from src.data.preprocessing.errors import PreprocessingError
from src.data.preprocessing.features import FeatureBuilder
from src.data.preprocessing.masking import Masker
from src.data.preprocessing.metadata import PREPROCESSING_VERSION, PreprocessingMetadata
from src.data.preprocessing.missing import MissingValueHandler
from src.data.preprocessing.normalization import Normalizer
from src.data.preprocessing.spatial import SpatialAligner
from src.data.preprocessing.splits import (
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
    TemporalSplit,
    assert_fit_window,
    check_disjoint,
    split_summary,
)
from src.data.preprocessing.temporal import TemporalAligner
from src.data.preprocessing.tensors import TensorAssembler, TensorBundle

PathLike = Union[str, Path]

#: The step after which the chronological split is decided. Everything up
#: to and including this one runs on the full dataset; the split boundary
#: is then derived from the resulting (aligned) time axis.
SPLIT_AFTER_STEP = "temporal_alignment"

#: Chronological order the periods are concatenated back in.
PERIOD_ORDER: Tuple[str, ...] = ("train", "val", "test")

#: Default file names written by `PreprocessingResult.save`.
TENSOR_FILENAME = "neer_tensors.npz"
METADATA_FILENAME = "neer_preprocessing_metadata.json"
REPORT_FILENAME = "neer_preprocessing_report.txt"


def concat_along_time(datasets: Sequence[OceanDataset]) -> OceanDataset:
    """Join period datasets back into one, in the order given.

    Variables with a time axis are concatenated along it. Variables
    without one — the land mask, the static geographic features — are
    identical in every period by construction, so the first period's copy
    is kept and the rest are checked for agreement rather than trusted.
    """
    datasets = [d for d in datasets if d.n_time]
    if not datasets:
        raise PreprocessingError("nothing to concatenate: every period is empty")
    if len(datasets) == 1:
        return datasets[0]

    first = datasets[0]
    names = set(first.variables)
    for other in datasets[1:]:
        if set(other.variables) != names:
            raise PreprocessingError(
                "periods carry different variables and cannot be concatenated: "
                f"{sorted(names ^ set(other.variables))}"
            )

    variables = {}
    for name, variable in first.variables.items():
        axis = axis_of(variable, "time")
        if axis is None:
            for other in datasets[1:]:
                if not np.array_equal(
                    np.asarray(other[name].values), np.asarray(variable.values)
                ):
                    raise PreprocessingError(
                        f"time-independent variable '{name}' differs between periods; "
                        "it should be identical in all of them"
                    )
            variables[name] = variable
            continue
        joined = np.concatenate(
            [np.asarray(d[name].values) for d in datasets], axis=axis
        )
        variables[name] = with_values(variable, joined)

    coords = dict(first.coords)
    coords["time"] = np.concatenate([np.asarray(d.coords["time"]) for d in datasets])
    return with_variables(first, variables, coords=coords)


def default_steps(**overrides: Any) -> List[PreprocessingStep]:
    """The seven standard NEER preprocessing steps, in pipeline order.

    `overrides` takes a step name mapped to an already-constructed step,
    so a caller can swap one stage without restating the rest::

        default_steps(normalization=Normalizer(method="robust"))
    """
    steps: List[PreprocessingStep] = [
        CoordinateNormalizer(),
        TemporalAligner(),
        SpatialAligner(),
        MissingValueHandler(),
        Masker(),
        FeatureBuilder(),
        Normalizer(),
    ]
    if not overrides:
        return steps
    by_name = {step.name: index for index, step in enumerate(steps)}
    for name, replacement in overrides.items():
        if name not in by_name:
            raise PreprocessingError(
                f"unknown step {name!r}; the pipeline's steps are {list(by_name)}"
            )
        steps[by_name[name]] = replacement
    return steps


@dataclass(frozen=True)
class PreprocessingResult:
    """Everything one pipeline run produced."""

    dataset: OceanDataset
    tensors: TensorBundle
    split: TemporalSplit
    metadata: PreprocessingMetadata
    checks: List[str] = field(default_factory=list)

    # -- per-period views --------------------------------------------------

    @property
    def train(self) -> TensorBundle:
        return self.tensors.split("train")

    @property
    def val(self) -> TensorBundle:
        return self.tensors.split("val")

    @property
    def test(self) -> TensorBundle:
        return self.tensors.split("test")

    def period(self, name: str) -> TensorBundle:
        """The tensors for one period ('train', 'val' or 'test')."""
        return self.tensors.split(name)

    # -- persistence -------------------------------------------------------

    def save(
        self,
        directory: PathLike,
        *,
        tensor_filename: str = TENSOR_FILENAME,
        metadata_filename: str = METADATA_FILENAME,
        report_filename: Optional[str] = REPORT_FILENAME,
        compress: bool = True,
    ) -> Dict[str, Path]:
        """Write tensors, metadata and a readable report into `directory`.

        The metadata records the tensor file it describes, so the two can
        be matched up later even if they are moved apart.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tensor_path = self.tensors.save(directory / tensor_filename, compress=compress)
        self.metadata.outputs = {
            "tensors": str(tensor_path),
            "tensor_bytes": int(tensor_path.stat().st_size),
            "metadata": str(directory / metadata_filename),
        }
        metadata_path = self.metadata.save(directory / metadata_filename)

        written = {"tensors": tensor_path, "metadata": metadata_path}
        if report_filename:
            report_path = directory / report_filename
            report_path.write_text(self.metadata.to_text(), encoding="utf-8")
            written["report"] = report_path
        return written

    def describe(self) -> str:
        """A short human-readable summary of the run."""
        return self.metadata.to_text()


class PreprocessingPipeline:
    """Run the NEER preprocessing steps in order, without leaking test data."""

    def __init__(
        self,
        steps: Optional[Sequence[PreprocessingStep]] = None,
        *,
        split: Optional[TemporalSplit] = None,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        test_fraction: float = DEFAULT_TEST_FRACTION,
        assembler: Optional[TensorAssembler] = None,
        environment: Optional[str] = None,
        name: str = "neer-preprocessing",
    ) -> None:
        self.steps: List[PreprocessingStep] = list(
            steps if steps is not None else default_steps()
        )
        self.split = split
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.assembler = assembler if assembler is not None else TensorAssembler()
        self.environment = environment
        self.name = name
        self._fitted_split: Optional[TemporalSplit] = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: Any = None, *, environment: Optional[str] = None, **kwargs: Any
    ) -> "PreprocessingPipeline":
        """Build a pipeline from the `preprocessing:` section of a NEER config.

        Every key is optional; anything absent falls back to the step's
        own default, so `configs/base.yaml` only has to state what the
        project actually wants to pin down.
        """
        if config is None:
            from src.utils.config import load_config

            config = load_config(environment)

        raw = getattr(config, "raw", config) or {}
        settings = dict(raw.get("preprocessing") or {})
        environment = environment or getattr(config, "environment", None)

        def section(key: str) -> Dict[str, Any]:
            return dict(settings.get(key) or {})

        steps = default_steps(
            coordinate_normalization=CoordinateNormalizer(**section("coordinates")),
            temporal_alignment=TemporalAligner(**section("temporal")),
            spatial_alignment=SpatialAligner(environment=environment, **section("spatial")),
            missing_values=MissingValueHandler(**section("missing")),
            masking=Masker(**section("masking")),
            feature_construction=FeatureBuilder(**section("features")),
            normalization=Normalizer(**section("normalization")),
        )

        split_settings = section("split")
        return cls(
            steps,
            val_fraction=split_settings.get("val_fraction", DEFAULT_VAL_FRACTION),
            test_fraction=split_settings.get("test_fraction", DEFAULT_TEST_FRACTION),
            assembler=TensorAssembler(**section("tensors")),
            environment=environment,
            **kwargs,
        )

    # -- inspection --------------------------------------------------------

    def step(self, name: str) -> PreprocessingStep:
        """One step by name, e.g. `pipeline.step("normalization")`."""
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"no step '{name}'; available: {[s.name for s in self.steps]}")

    @property
    def learned_steps(self) -> List[PreprocessingStep]:
        """The steps that hold parameters derived from data."""
        return [step for step in self.steps if step.learns_from_data]

    @property
    def fitted(self) -> bool:
        """True once every learned step has been fitted."""
        return all(step.fitted for step in self.steps)

    def describe(self) -> str:
        lines = [f"{self.name} ({len(self.steps)} steps)"]
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"  {index}. {step.describe()}")
        lines.append(f"  {len(self.steps) + 1}. {self.assembler.name} [stateless]")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<PreprocessingPipeline steps={[s.name for s in self.steps]} fitted={self.fitted}>"

    # -- the run -----------------------------------------------------------

    def run(
        self,
        dataset: OceanDataset,
        *,
        split: Optional[TemporalSplit] = None,
    ) -> PreprocessingResult:
        """Fit and apply the whole pipeline, returning tensors and metadata.

        Stages 1-2 run on the full dataset (they are per-timestep
        coordinate and calendar arithmetic). The split is then derived
        from the aligned time axis, and stages 3-7 run on each period
        separately, with the learned steps fitted on the training period
        only. The periods are concatenated for tensor assembly.

        Parameters
        ----------
        split:
            An explicit chronological split. When omitted, one is derived
            from the aligned time axis using `val_fraction` /
            `test_fraction`.
        """
        checks: List[str] = []
        source_summary = dataset.summary()

        # -- Stages 1-2: stateless, run before the split is decided --------
        index = self._split_boundary()
        working = dataset
        for step in self.steps[:index]:
            working = self._apply(step, working)

        # -- The split, derived from the aligned time axis -----------------
        times = working.coords.get("time")
        if times is None or times.size == 0:
            raise PreprocessingError(
                "the pipeline needs a time axis to split on; the dataset has none "
                "after temporal alignment"
            )
        active_split = split or self.split or TemporalSplit.from_fractions(
            times, val_fraction=self.val_fraction, test_fraction=self.test_fraction
        )
        ok, message = check_disjoint(active_split, times)
        if not ok:
            raise PreprocessingError(f"invalid split: {message}")
        checks.append(f"split: {message}")
        self._fitted_split = active_split

        # -- Stages 3-7: fit on training data, transform periods apart -----
        #
        # Each period is carried through the remaining steps in its own
        # dataset. Nothing downstream ever sees two periods in one array,
        # so no step can interpolate, aggregate or normalize across a
        # split boundary.
        periods: Dict[str, OceanDataset] = {
            period: select_times(working, mask)
            for period, mask in active_split.masks(times).items()
            if mask.any()
        }
        if "train" not in periods:  # pragma: no cover - check_disjoint catches this
            raise PreprocessingError("the training period is empty; nothing to fit on")

        fit_records: List[Dict[str, Any]] = []
        for step in self.steps[index:]:
            if step.learns_from_data:
                fit_records.append(self._fit_on_training(step, periods["train"], active_split))
            periods = {
                period: self._apply(step, period_dataset)
                for period, period_dataset in periods.items()
            }

        checks.append(
            f"fit window: {len(fit_records)} learned step(s) fitted on training data only"
        )
        checks.append(
            "isolation: post-split stages ran on "
            f"{', '.join(sorted(periods))} independently"
        )

        working = concat_along_time([periods[p] for p in PERIOD_ORDER if p in periods])

        # -- Stage 8: tensors ----------------------------------------------
        bundle = self.assembler.assemble(working, split=active_split)
        checks.extend(self._check_tensors(bundle))

        metadata = self._metadata(
            source_summary=source_summary,
            dataset=dataset,
            split=active_split,
            times=times,
            bundle=bundle,
            fit_records=fit_records,
            checks=checks,
        )
        return PreprocessingResult(
            dataset=working,
            tensors=bundle,
            split=active_split,
            metadata=metadata,
            checks=checks,
        )

    def transform(self, dataset: OceanDataset) -> OceanDataset:
        """Apply the already-fitted pipeline to new data, learning nothing.

        This is the inference path: the same coordinate conventions, the
        same climatology, the same normalization statistics that the
        training run produced, applied to data the pipeline has never
        seen. Learned steps raise `NotFittedError` if `run` has not been
        called, which is the correct failure — silently refitting on the
        new data is the bug this whole module exists to prevent.
        """
        working = dataset
        for step in self.steps:
            working = step.transform(working)
        return working

    def assemble(self, dataset: OceanDataset) -> TensorBundle:
        """`transform` a dataset and pack the result into tensors."""
        return self.assembler.assemble(self.transform(dataset))

    # -- internals ---------------------------------------------------------

    def _split_boundary(self) -> int:
        """Index of the first step that runs *after* the split is decided."""
        for position, step in enumerate(self.steps):
            if step.name == SPLIT_AFTER_STEP:
                return position + 1
        # No temporal aligner configured: split on the time axis as given.
        for position, step in enumerate(self.steps):
            if step.learns_from_data:
                return position
        return len(self.steps)

    @staticmethod
    def _apply(step: PreprocessingStep, dataset: OceanDataset) -> OceanDataset:
        result = step.transform(dataset)
        if not isinstance(result, OceanDataset):  # pragma: no cover - defensive
            raise PreprocessingError(
                f"step '{step.name}' returned {type(result).__name__}, not an OceanDataset"
            )
        return result

    def _fit_on_training(
        self, step: PreprocessingStep, training: OceanDataset, split: TemporalSplit
    ) -> Dict[str, Any]:
        """Verify `training` really is training data, then fit `step` on it.

        The guard reads the timestamps actually present in the dataset it
        is about to fit on — not the ones the caller intended to pass —
        so a bug in how the periods were sliced cannot slip through.
        """
        assert_fit_window(split, training.coords["time"], step=step.name)
        step.fit(training)

        train_times = training.coords["time"]
        return {
            "step": step.name,
            "n_timesteps": int(train_times.size),
            "first": str(train_times.min()),
            "last": str(train_times.max()),
            "guard": "assert_fit_window",
        }

    @staticmethod
    def _check_tensors(bundle: TensorBundle) -> List[str]:
        """Post-conditions the tensors must satisfy to be model-ready."""
        checks = []
        if not np.all(np.isfinite(bundle.inputs)):
            raise PreprocessingError(
                "assembled input tensor still contains non-finite values; the "
                "missing-value stage and the assembler's fill_value should make that "
                "impossible"
            )
        checks.append("tensors: inputs are finite everywhere")

        if bundle.targets is not None:
            if not np.all(np.isfinite(bundle.targets)):
                raise PreprocessingError("assembled target tensor contains non-finite values")
            checks.append("tensors: targets are finite everywhere")
        if bundle.target_mask is not None and not bundle.target_mask.any():
            raise PreprocessingError(
                "no target cell is marked observed; the model would have nothing to "
                "compute a loss on"
            )
        counts = {k: int(np.asarray(v).sum()) for k, v in bundle.split_masks.items()}
        checks.append(f"tensors: split sizes {counts}")
        return checks

    def _metadata(
        self,
        *,
        source_summary: Dict[str, Any],
        dataset: OceanDataset,
        split: TemporalSplit,
        times: np.ndarray,
        bundle: TensorBundle,
        fit_records: List[Dict[str, Any]],
        checks: List[str],
    ) -> PreprocessingMetadata:
        """Assemble the full record of this run."""
        split_meta = split_summary(split, times)
        split_meta["spatial_coverage"] = bundle.spatial_coverage()

        notes: List[str] = []
        if dataset.is_synthetic:
            notes.append(
                "SYNTHETIC DATA: these tensors derive from the NEER demo dataset and "
                "must not be used for any scientific purpose or presented as results."
            )
            disclaimer = dataset.attrs.get("disclaimer")
            if disclaimer:
                notes.append(str(disclaimer))

        project: Dict[str, Any] = {}
        try:  # metadata should never be the reason a run fails
            from src.utils.config import load_config

            config = load_config(self.environment)
            project = {
                "name": config.project.name,
                "problem_id": config.project.problem_id,
                "organization": config.project.organization,
            }
        except Exception:  # pragma: no cover - config problems surface elsewhere
            project = {"name": "NEER", "problem_id": "SIH26066"}

        return PreprocessingMetadata(
            pipeline=self.name,
            version=PREPROCESSING_VERSION,
            project=project,
            environment=self.environment,
            source={
                **source_summary,
                "data_mode": dataset.data_mode,
                "is_synthetic": dataset.is_synthetic,
            },
            steps=[step.to_dict() for step in self.steps] + [self.assembler.to_dict()],
            split=split_meta,
            tensors=bundle.summary(),
            leakage=json_safe(
                {
                    "policy": (
                        "Chronological split. (1) Steps that learn from data are fitted "
                        "on the training period only. (2) After the split, each period "
                        "is transformed in its own dataset, so no step can interpolate "
                        "or aggregate across a split boundary."
                    ),
                    "guard": "src.data.preprocessing.splits.assert_fit_window",
                    "split_decided_after": SPLIT_AFTER_STEP,
                    "learned_steps": fit_records,
                    "stateless_steps": [
                        step.name for step in self.steps if not step.learns_from_data
                    ],
                    "periods_transformed_independently": list(PERIOD_ORDER),
                    "checks": checks,
                }
            ),
            notes=notes,
        )
