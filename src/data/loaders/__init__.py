"""
NEER — Neural Embedding based Estimation and Reconstruction

Package: src/data/loaders
SIH Problem Statement: SIH26066
Organization: MoES / INCOIS

The data loading layer: reusable loaders that turn NetCDF, CSV, and
NumPy-compatible sources into one standardized internal representation,
`OceanDataset`.

    from src.data.loaders import load_demo_dataset, load_netcdf, load_csv

    dataset = load_demo_dataset()            # synthetic demo data
    dataset = load_netcdf("argo.nc")         # real gridded products
    dataset = load_csv("profiles.csv")       # point/profile observations

Whatever the source, the result is the same kind of object with the same
guarantees — canonical coordinate names (`time`, `depth`, `lat`, `lon`),
canonical axis order, `NaN` for missing data, canonical units, and a
validation report attached. See `representation.py` for the full contract
and `validation.py` for what is checked.

`load_demo_dataset` returns synthetic data (`data_mode="DEMO_SYNTHETIC"`,
`dataset.is_synthetic is True`) and anything reporting results to a user
should say so.

No model training happens here — this layer only reads, standardizes, and
validates data.
"""

from src.data.loaders.csv_loader import load_csv
from src.data.loaders.demo_loader import (
    DEFAULT_DEMO_DATASET,
    DEFAULT_DEMO_METADATA,
    DEMO_DATA_MODE,
    demo_dataset_exists,
    load_demo_dataset,
)
from src.data.loaders.errors import (
    DataValidationError,
    LoaderError,
    MissingDependencyError,
    SchemaError,
    UnsupportedFormatError,
)
from src.data.loaders.netcdf_loader import load_netcdf, save_netcdf
from src.data.loaders.numpy_loader import load_npy, load_npz
from src.data.loaders.representation import OceanDataset, Variable, build_variable
from src.data.loaders.schema import (
    CANONICAL_COORDS,
    CANONICAL_DIM_ORDER,
    VARIABLE_SPECS,
    VariableSpec,
    canonical_coord_name,
    canonical_variable_name,
    lookup_spec,
)
from src.data.loaders.units import convert, convert_to_canonical, normalize_unit
from src.data.loaders.validation import (
    ALL_CHECKS,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_dataset,
)

__all__ = [
    # Loaders
    "load_netcdf",
    "load_csv",
    "load_demo_dataset",
    "load_npz",
    "load_npy",
    "save_netcdf",
    "demo_dataset_exists",
    # Representation
    "OceanDataset",
    "Variable",
    "build_variable",
    # Schema
    "CANONICAL_COORDS",
    "CANONICAL_DIM_ORDER",
    "VARIABLE_SPECS",
    "VariableSpec",
    "canonical_coord_name",
    "canonical_variable_name",
    "lookup_spec",
    # Units
    "normalize_unit",
    "convert",
    "convert_to_canonical",
    # Validation
    "validate_dataset",
    "ValidationReport",
    "ValidationIssue",
    "Severity",
    "ALL_CHECKS",
    # Errors
    "LoaderError",
    "MissingDependencyError",
    "UnsupportedFormatError",
    "SchemaError",
    "DataValidationError",
    # Demo constants
    "DEFAULT_DEMO_DATASET",
    "DEFAULT_DEMO_METADATA",
    "DEMO_DATA_MODE",
]
