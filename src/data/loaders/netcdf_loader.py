"""
NetCDF loading for NEER (`load_netcdf`).

NetCDF is the format real INCOIS/Copernicus/ARGO products arrive in, so
this is the loader that will carry the project once actual data is wired
up. It is built on xarray, which handles CF conventions (time decoding,
`scale_factor`/`add_offset`, compression) far better than hand-rolled
netCDF4 calls would.

xarray and a NetCDF engine are *optional* dependencies: they are imported
lazily so that the CSV/NumPy/demo paths — and everything that imports
them, including the backend — keep working without them. Calling
`load_netcdf` without xarray installed raises `MissingDependencyError`
with the exact install command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from src.data.loaders._common import (
    finalize_dataset,
    select_variables,
    standardize_coords,
    standardize_variable,
)
from src.data.loaders._xarray import require_netcdf_engine, require_xarray
from src.data.loaders.errors import SchemaError
from src.data.loaders.representation import OceanDataset
from src.data.loaders.validation import DEFAULT_MISSING_WARN_FRACTION

PathLike = Union[str, Path]

NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf", ".netcdf")


def load_netcdf(
    path: PathLike,
    *,
    variables: Optional[Sequence[str]] = None,
    rename: Optional[Mapping[str, str]] = None,
    engine: Optional[str] = None,
    decode_times: bool = True,
    standardize_names: bool = True,
    normalize_units: bool = True,
    replace_sentinels: bool = True,
    validate: bool = True,
    strict: bool = False,
    check_domain: bool = True,
    domain: Any = None,
    missing_warn_fraction: float = DEFAULT_MISSING_WARN_FRACTION,
    open_kwargs: Optional[Mapping[str, Any]] = None,
) -> OceanDataset:
    """Load a NetCDF file into NEER's standardized representation.

    Parameters
    ----------
    path:
        Path to a `.nc` / `.nc4` / `.cdf` file.
    variables:
        Which variables to load, by NEER name or by the file's own name.
        Defaults to every data variable in the file.
    rename:
        Extra `{source_name: neer_name}` mappings applied before
        standardization, for variables the registry cannot resolve on its
        own (a file whose temperature field is just called `temp`, say).
    engine:
        xarray engine. Defaults to the best available of `netcdf4`,
        `h5netcdf`, `scipy`.
    decode_times:
        Let xarray decode CF time units. If decoding fails, the file is
        reopened with `decode_times=False` and the raw time values are
        kept — validation then reports that time could not be decoded,
        rather than the load failing outright.
    standardize_names / normalize_units / replace_sentinels:
        Stages of the standardization pipeline; see `_common.py`.
    validate / strict / check_domain / domain / missing_warn_fraction:
        Validation behaviour; see `validation.validate_dataset`.

    Returns
    -------
    OceanDataset

    Raises
    ------
    MissingDependencyError
        xarray or a NetCDF engine is not installed.
    FileNotFoundError
        The file does not exist.
    SchemaError
        The file holds no data variables.
    DataValidationError
        Validation failed (or, with `strict=True`, produced warnings).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    xr = require_xarray("reading NetCDF files")
    engine = engine or require_netcdf_engine()

    open_kwargs = dict(open_kwargs or {})
    open_kwargs.setdefault("engine", engine)

    decode_failed = False

    try:
        source = xr.open_dataset(
            path,
            decode_times=decode_times,
            **open_kwargs,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        if not decode_times:
            raise

        # Out-of-range or non-standard calendars are common in older
        # products; keep the raw values rather than failing the load.
        decode_failed = True
        _decode_error = exc

        source = xr.open_dataset(
            path,
            decode_times=False,
            **open_kwargs,
        )

    with source as raw:
        if rename:
            present = {
                k: v
                for k, v in rename.items()
                if k in raw.variables
            }

            if present:
                raw = raw.rename(present)

        available = [
            str(name)
            for name in raw.data_vars
        ]

        if not available:
            raise SchemaError(
                f"{path} contains no data variables "
                f"(coordinates found: "
                f"{sorted(str(c) for c in raw.coords)})"
            )

        selected = select_variables(
            available,
            variables,
            source=str(path),
        )

        coords = {
            str(name): np.asarray(coord.values)
            for name, coord in raw.coords.items()
            if coord.ndim == 1
        }

        coords = standardize_coords(coords)

        built = {}

        for name in selected:
            array = raw[name]

            attrs = dict(array.attrs)

            units = attrs.pop("units", None)

            variable = standardize_variable(
                name,
                np.asarray(array.values),
                tuple(str(d) for d in array.dims),
                units=units,
                attrs=attrs,
                standardize_names=standardize_names,
                replace_sentinels=replace_sentinels,
                normalize_units=normalize_units,
            )

            built[variable.name] = variable

        dataset_attrs = dict(raw.attrs)

    if decode_failed:
        dataset_attrs["time_decoding_error"] = str(_decode_error)

    # Restore dictionary attributes that were serialized as JSON
    # by save_netcdf().
    for key, value in list(dataset_attrs.items()):
        if isinstance(value, str):
            try:
                decoded = json.loads(value)

                if isinstance(decoded, dict):
                    dataset_attrs[key] = decoded

            except (json.JSONDecodeError, TypeError):
                pass

    dataset = OceanDataset(
        variables=built,
        coords=coords,
        attrs=dataset_attrs,
        source=str(path),
        source_format="netcdf",
    )

    return finalize_dataset(
        dataset,
        validate=validate,
        strict=strict,
        domain=domain,
        check_domain=check_domain,
        missing_warn_fraction=missing_warn_fraction,
    )


def save_netcdf(
    dataset: OceanDataset,
    path: PathLike,
    **to_netcdf_kwargs: Any,
) -> Path:
    """Write an `OceanDataset` back out as NetCDF.

    Mostly used by tests and by scripts that need to hand a NEER dataset
    to an external tool; the loading layer itself never calls this.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    require_xarray("writing NetCDF files")

    to_netcdf_kwargs.setdefault(
        "engine",
        require_netcdf_engine("writing NetCDF files"),
    )

    xr_dataset = dataset.to_xarray()

    # NetCDF attributes cannot contain Python dictionaries or booleans.
    # Serialize dictionary attributes as JSON strings and booleans as integers before writing.
    def _sanitize_attrs(attrs: dict) -> None:
        for key, value in list(attrs.items()):
            if isinstance(value, dict):
                attrs[key] = json.dumps(value)
            elif isinstance(value, (bool, np.bool_)):
                attrs[key] = int(value)

    _sanitize_attrs(xr_dataset.attrs)
    for var in xr_dataset.variables.values():
        _sanitize_attrs(var.attrs)

    xr_dataset.to_netcdf(
        path,
        **to_netcdf_kwargs,
    )

    return path