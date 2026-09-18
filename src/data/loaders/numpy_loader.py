"""
NumPy-compatible loading for NEER (`load_npz`, `load_npy`).

`.npz` is the project's own interchange format: `scripts/create_demo_data.py`
writes the demo dataset as one, and cached/preprocessed arrays are cheapest
to keep this way. The format carries no schema at all — just named arrays —
so this loader reconstructs the missing metadata from three sources, in
order of preference:

1. an explicit `dims=` / `units=` argument from the caller,
2. a sidecar metadata JSON (`metadata=` or `<name>_metadata.json`), which
   is how the demo dataset records dims and units,
3. shape inference against the coordinate arrays, which only succeeds when
   it is unambiguous (see `schema.infer_dims_from_shape`).

Arrays whose dims cannot be established are skipped rather than guessed at,
and reported in `attrs["skipped_variables"]`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from src.data.loaders._common import (
    finalize_dataset,
    select_variables,
    standardize_coords,
    standardize_variable,
)
from src.data.loaders.errors import SchemaError
from src.data.loaders.representation import OceanDataset
from src.data.loaders.schema import (
    CANONICAL_COORDS,
    canonical_coord_name,
    infer_dims_from_shape,
)
from src.data.loaders.validation import DEFAULT_MISSING_WARN_FRACTION

PathLike = Union[str, Path]

NUMPY_SUFFIXES = (".npz", ".npy")


def _split_coords_and_variables(arrays: Mapping[str, np.ndarray]) -> tuple:
    """Separate 1D coordinate arrays from data variables by canonical name."""
    coords: Dict[str, np.ndarray] = {}
    variables: Dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        canonical = canonical_coord_name(name)
        if canonical in CANONICAL_COORDS and np.ndim(array) == 1:
            coords[canonical] = np.asarray(array)
        else:
            variables[str(name)] = np.asarray(array)
    return coords, variables


def _dims_from_metadata(metadata: Optional[Mapping[str, Any]], name: str) -> Optional[Sequence[str]]:
    if not metadata:
        return None
    entry = (metadata.get("variables") or {}).get(name)
    if isinstance(entry, Mapping) and entry.get("dims"):
        return list(entry["dims"])
    return None


def _units_from_metadata(metadata: Optional[Mapping[str, Any]], name: str) -> Optional[str]:
    if not metadata:
        return None
    entry = (metadata.get("variables") or {}).get(name)
    if isinstance(entry, Mapping) and entry.get("units"):
        return str(entry["units"])
    return None


def _default_metadata_path(path: Path) -> Optional[Path]:
    """Look for a sidecar metadata JSON next to `path`."""
    candidates = [
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}_metadata.json"),
        path.with_name(f"{path.stem.replace('_dataset', '')}_metadata.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_npz(
    path: PathLike,
    *,
    variables: Optional[Sequence[str]] = None,
    dims: Optional[Mapping[str, Sequence[str]]] = None,
    units: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    metadata_path: Optional[PathLike] = None,
    use_sidecar_metadata: bool = True,
    attrs: Optional[Mapping[str, Any]] = None,
    standardize_names: bool = True,
    normalize_units: bool = True,
    replace_sentinels: bool = True,
    allow_pickle: bool = False,
    validate: bool = True,
    strict: bool = False,
    check_domain: bool = True,
    domain: Any = None,
    missing_warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
    source_format: str = "npz",
) -> OceanDataset:
    """Load a `.npz` archive of named arrays into an `OceanDataset`.

    Arrays named after a coordinate (`lat`, `latitude`, `time`, `depth`, ...)
    and 1-dimensional become coordinates; everything else becomes a variable.

    Parameters
    ----------
    dims:
        `{variable: (dim, ...)}` for arrays whose dims cannot be inferred.
    units:
        `{variable: unit}`, overriding anything in the sidecar metadata.
    metadata / metadata_path / use_sidecar_metadata:
        A metadata mapping (or JSON file) in the shape written by
        `scripts/create_demo_data.py`: `{"variables": {name: {"dims": [...],
        "units": "..."}}, ...}`. Found automatically next to the `.npz`
        unless `use_sidecar_metadata=False`.

    Raises
    ------
    FileNotFoundError, SchemaError, DataValidationError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NumPy archive not found: {path}")

    resolved_metadata: Optional[Mapping[str, Any]] = metadata
    if resolved_metadata is None:
        candidate = Path(metadata_path) if metadata_path else (
            _default_metadata_path(path) if use_sidecar_metadata else None
        )
        if candidate is not None:
            if not candidate.exists():
                raise FileNotFoundError(f"metadata file not found: {candidate}")
            with open(candidate, "r", encoding="utf-8") as handle:
                resolved_metadata = json.load(handle)

    with np.load(path, allow_pickle=allow_pickle) as archive:
        arrays = {name: archive[name] for name in archive.files}

    if not arrays:
        raise SchemaError(f"{path} contains no arrays")

    raw_coords, raw_variables = _split_coords_and_variables(arrays)
    if not raw_variables:
        raise SchemaError(
            f"{path} contains only coordinate arrays ({sorted(raw_coords)}) and no variables"
        )

    coords = standardize_coords(raw_coords)
    coord_sizes = {name: int(values.size) for name, values in coords.items()}

    selected = select_variables(list(raw_variables), variables, source=str(path))

    explicit_dims = {str(k): tuple(v) for k, v in (dims or {}).items()}
    built: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}

    for name in selected:
        array = raw_variables[name]
        variable_dims = (
            explicit_dims.get(name)
            or _dims_from_metadata(resolved_metadata, name)
            or None
        )
        if variable_dims is None:
            try:
                variable_dims = infer_dims_from_shape(
                    array.shape, coord_sizes, variable=name
                )
            except ValueError as exc:
                skipped[name] = str(exc)
                continue

        if len(variable_dims) != array.ndim:
            raise SchemaError(
                f"{path}: dims {tuple(variable_dims)} given for '{name}' do not match its "
                f"shape {array.shape}"
            )

        variable = standardize_variable(
            name,
            array,
            variable_dims,
            units=(units or {}).get(name) or _units_from_metadata(resolved_metadata, name),
            attrs={},
            standardize_names=standardize_names,
            replace_sentinels=replace_sentinels and array.dtype != bool,
            normalize_units=normalize_units,
        )
        built[variable.name] = variable

    if not built:
        raise SchemaError(
            f"{path}: could not determine dimensions for any variable "
            f"({', '.join(f'{k}: {v}' for k, v in skipped.items())}). Pass dims=..."
        )

    dataset_attrs: Dict[str, Any] = {}
    if resolved_metadata:
        # Keep provenance (data_mode, disclaimer, seed, generator, ...) but
        # not the bulky per-variable block, which is now on the Variables.
        dataset_attrs.update(
            {k: v for k, v in resolved_metadata.items() if k not in ("variables",)}
        )
    dataset_attrs.update(dict(attrs or {}))
    if skipped:
        dataset_attrs["skipped_variables"] = skipped

    dataset = OceanDataset(
        variables=built,
        coords=coords,
        attrs=dataset_attrs,
        source=str(path),
        source_format=source_format,
    )

    return finalize_dataset(
        dataset,
        validate=validate,
        strict=strict,
        domain=domain,
        check_domain=check_domain,
        missing_warn_fraction=missing_warn_fraction,
    )


def load_npy(
    path: PathLike,
    *,
    name: str,
    dims: Sequence[str],
    coords: Optional[Mapping[str, np.ndarray]] = None,
    units: Optional[str] = None,
    attrs: Optional[Mapping[str, Any]] = None,
    validate: bool = True,
    strict: bool = False,
    check_domain: bool = True,
    domain: Any = None,
    **kwargs: Any,
) -> OceanDataset:
    """Load a single `.npy` array as a one-variable dataset.

    A bare `.npy` carries no names, dims, or coordinates, so `name` and
    `dims` are required and `coords` should be supplied for validation to
    be meaningful.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NumPy array not found: {path}")

    array = np.load(path, allow_pickle=kwargs.pop("allow_pickle", False))
    variable = standardize_variable(
        name,
        array,
        dims,
        units=units,
        attrs={},
        standardize_names=kwargs.pop("standardize_names", True),
        replace_sentinels=kwargs.pop("replace_sentinels", True),
        normalize_units=kwargs.pop("normalize_units", True),
    )

    dataset = OceanDataset(
        variables={variable.name: variable},
        coords=standardize_coords(coords or {}),
        attrs=dict(attrs or {}),
        source=str(path),
        source_format="npy",
    )
    return finalize_dataset(
        dataset,
        validate=validate,
        strict=strict,
        domain=domain,
        check_domain=check_domain,
    )
