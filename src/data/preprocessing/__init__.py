"""
NEER preprocessing — raw ocean data to model-ready normalized tensors.

Package: src/data/preprocessing
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

The pipeline
------------
::

    raw data
      -> coordinate normalization   CoordinateNormalizer
      -> temporal alignment         TemporalAligner
      -> spatial alignment          SpatialAligner
      -> missing-value handling     MissingValueHandler   (learned)
      -> masking                    Masker
      -> feature construction       FeatureBuilder
      -> normalization              Normalizer            (learned)
      -> tensor assembly            TensorAssembler
    normalized tensors + metadata

Usage
-----
The whole thing, on the demo dataset::

    >>> from src.data.loaders import load_demo_dataset
    >>> from src.data.preprocessing import PreprocessingPipeline
    >>> pipeline = PreprocessingPipeline.from_config()
    >>> result = pipeline.run(load_demo_dataset())
    >>> result.tensors.inputs.shape
    (24, 11, 101, 241)
    >>> result.save("data/processed")

Or one stage on its own — every step is a standalone, reusable object::

    >>> from src.data.preprocessing import CoordinateNormalizer
    >>> dataset = CoordinateNormalizer(lon_convention="-180-180").transform(dataset)

Two things this package guarantees
----------------------------------
**Nothing is learned from validation or test data.** Steps declare
whether they learn (`learns_from_data`); the pipeline fits those on the
training period only, and `assert_fit_window` re-checks the timestamps it
was actually handed. See `pipeline.py` for the full argument.

**Every output is traceable.** `PreprocessingMetadata` records each
step's configuration, fitted state and per-run report alongside the
tensors, including the normalization statistics needed to put
predictions back into physical units.
"""

from src.data.preprocessing.base import LearnedStep, PreprocessingStep, StatelessStep
from src.data.preprocessing.channels import (
    CHANNEL_DESCRIPTIONS,
    NEER_CHANNEL_ORDER,
    NEER_N_CHANNELS,
    validate_channel_order,
)
from src.data.preprocessing.climatology import (
    CLIMATOLOGY_VERSION,
    DEFAULT_VARIABLES as CLIMATOLOGY_VARIABLES,
    MonthlyClimatology,
    compute_delta_temperature,
    reconstruct_temperature,
)
from src.data.preprocessing.coordinates import (
    LON_CONVENTIONS,
    CoordinateNormalizer,
    to_convention,
)
from src.data.preprocessing.errors import (
    LeakageError,
    NotFittedError,
    PreprocessingError,
    StepConfigurationError,
)
from src.data.preprocessing.features import (
    AVAILABLE_FEATURES,
    DEFAULT_FEATURES,
    FeatureBuilder,
)
from src.data.preprocessing.masking import OCEAN_MASK, Masker
from src.data.preprocessing.metadata import (
    PREPROCESSING_VERSION,
    PreprocessingMetadata,
)
from src.data.preprocessing.missing import OBSERVED_SUFFIX, MissingValueHandler
from src.data.preprocessing.normalization import METHODS as NORMALIZATION_METHODS
from src.data.preprocessing.normalization import Normalizer
from src.data.preprocessing.pipeline import (
    METADATA_FILENAME,
    TENSOR_FILENAME,
    PreprocessingPipeline,
    PreprocessingResult,
    default_steps,
)
from src.data.preprocessing.spatial import SpatialAligner, interp_axis
from src.data.preprocessing.splits import (
    TemporalSplit,
    assert_fit_window,
    check_disjoint,
    split_summary,
)
from src.data.preprocessing.temporal import FREQUENCIES, TemporalAligner, snap_times
from src.data.preprocessing.tensors import TensorAssembler, TensorBundle

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "PreprocessingResult",
    "default_steps",
    "TENSOR_FILENAME",
    "METADATA_FILENAME",
    # Steps, in pipeline order
    "CoordinateNormalizer",
    "TemporalAligner",
    "SpatialAligner",
    "MissingValueHandler",
    "Masker",
    "FeatureBuilder",
    "Normalizer",
    "TensorAssembler",
    # Step contract
    "PreprocessingStep",
    "StatelessStep",
    "LearnedStep",
    # Splits & the leakage guard
    "TemporalSplit",
    "assert_fit_window",
    "check_disjoint",
    "split_summary",
    # Outputs
    "TensorBundle",
    "PreprocessingMetadata",
    "PREPROCESSING_VERSION",
    # Helpers & constants
    "to_convention",
    "snap_times",
    "interp_axis",
    "LON_CONVENTIONS",
    "FREQUENCIES",
    "DEFAULT_FEATURES",
    "AVAILABLE_FEATURES",
    "NORMALIZATION_METHODS",
    "OCEAN_MASK",
    "OBSERVED_SUFFIX",
    # Phase 08 — the authoritative input-channel order
    "NEER_CHANNEL_ORDER",
    "NEER_N_CHANNELS",
    "CHANNEL_DESCRIPTIONS",
    "validate_channel_order",
    # Phase 11 — monthly climatology, delta_T and reconstruction
    "MonthlyClimatology",
    "compute_delta_temperature",
    "reconstruct_temperature",
    "CLIMATOLOGY_VARIABLES",
    "CLIMATOLOGY_VERSION",
    # Errors
    "PreprocessingError",
    "NotFittedError",
    "LeakageError",
    "StepConfigurationError",
]
