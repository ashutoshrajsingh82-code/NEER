"""
Demo dataset loading for NEER (`load_demo_dataset`).

Loads the synthetic dataset written by `scripts/create_demo_data.py` from
`data/demo/`. This is the dataset every other part of the project is
developed and tested against until real INCOIS/ARGO data is integrated,
so it is worth having a one-call entry point that needs no arguments:

    >>> from src.data.loaders import load_demo_dataset
    >>> dataset = load_demo_dataset()
    >>> dataset["sst"].shape
    (24, 101, 241)

⚠️ PROVENANCE
-------------
This data is **entirely synthetic** — closed-form formulas plus seeded
noise. It is not satellite data, not ARGO data, not reanalysis, and has
not been scientifically validated. The loader carries the generator's
`data_mode="DEMO_SYNTHETIC"` marker and its disclaimer into
`OceanDataset.attrs`, and refuses to load a file that lacks them unless
`require_demo_marker=False` is passed. `dataset.is_synthetic` is the
programmatic check; anything that reports results to a user should
consult it and label the output accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from src.data.loaders.errors import SchemaError
from src.data.loaders.numpy_loader import load_npz
from src.data.loaders.representation import OceanDataset
from src.data.loaders.validation import DEFAULT_MISSING_WARN_FRACTION

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DEFAULT_DEMO_DATASET = DEFAULT_DEMO_DIR / "neer_demo_ocean_dataset.npz"
DEFAULT_DEMO_METADATA = DEFAULT_DEMO_DIR / "neer_demo_metadata.json"

#: The marker `scripts/create_demo_data.py` writes into every metadata file.
DEMO_DATA_MODE = "DEMO_SYNTHETIC"

_REGENERATE_HINT = (
    "Generate it with:  python scripts/create_demo_data.py"
)


def load_demo_dataset(
    path: Optional[PathLike] = None,
    *,
    metadata_path: Optional[PathLike] = None,
    variables: Optional[Sequence[str]] = None,
    require_demo_marker: bool = True,
    validate: bool = True,
    strict: bool = False,
    check_domain: bool = True,
    domain: Any = None,
    missing_warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
    **npz_kwargs: Any,
) -> OceanDataset:
    """Load the synthetic NEER demo dataset.

    Parameters
    ----------
    path:
        Location of the `.npz`. Defaults to
        `data/demo/neer_demo_ocean_dataset.npz`.
    metadata_path:
        Location of the metadata JSON. Defaults to
        `data/demo/neer_demo_metadata.json` when it exists.
    variables:
        Subset of variables to load (e.g. `["sst", "sla"]`). Defaults to all.
    require_demo_marker:
        Refuse to load a file whose metadata does not declare
        `data_mode="DEMO_SYNTHETIC"`. Keeps a real dataset from being
        loaded — and later labelled — as demo data by mistake, and keeps
        synthetic data from losing its marker.

    Returns
    -------
    OceanDataset
        With `is_synthetic == True` and the generator's disclaimer in
        `attrs["disclaimer"]`.

    Raises
    ------
    FileNotFoundError
        The demo dataset has not been generated yet (the message includes
        the command that generates it).
    SchemaError
        The file loaded but does not carry the synthetic-data marker.
    """
    path = Path(path) if path is not None else DEFAULT_DEMO_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Demo dataset not found at {path}. {_REGENERATE_HINT}"
        )

    if metadata_path is not None:
        resolved_metadata = Path(metadata_path)
    elif path == DEFAULT_DEMO_DATASET and DEFAULT_DEMO_METADATA.exists():
        resolved_metadata = DEFAULT_DEMO_METADATA
    else:
        resolved_metadata = None

    dataset = load_npz(
        path,
        variables=variables,
        metadata_path=resolved_metadata,
        validate=validate,
        strict=strict,
        check_domain=check_domain,
        domain=domain,
        missing_warn_fraction=missing_warn_fraction,
        source_format="npz (demo)",
        **npz_kwargs,
    )

    if require_demo_marker and dataset.data_mode != DEMO_DATA_MODE:
        raise SchemaError(
            f"{path} does not declare data_mode={DEMO_DATA_MODE!r} "
            f"(found {dataset.data_mode!r}). The demo loader only loads the synthetic demo "
            "dataset, so downstream code can rely on is_synthetic being set. Pass "
            "require_demo_marker=False to override, or load it with load_npz() instead."
        )

    return dataset


def demo_dataset_exists(path: Optional[PathLike] = None) -> bool:
    """Whether the demo dataset has been generated yet."""
    return (Path(path) if path is not None else DEFAULT_DEMO_DATASET).exists()
