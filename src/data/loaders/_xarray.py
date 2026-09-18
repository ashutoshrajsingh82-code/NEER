"""
Lazy access to the optional xarray / NetCDF dependencies.

xarray (and a NetCDF engine behind it) is only needed for NetCDF I/O and
for `OceanDataset.to_xarray`. Importing it at module scope would make the
CSV, NumPy, and demo loaders — and everything that imports them, including
the backend — fail in an environment where xarray is not installed. The
helpers here defer the import to the moment it is genuinely required and
turn an `ImportError` into a `MissingDependencyError` with an actionable
message.
"""

from __future__ import annotations

from typing import Any

from src.data.loaders.errors import MissingDependencyError

#: Engines xarray can use to read NetCDF, in the order we prefer them.
NETCDF_ENGINES = ("netcdf4", "h5netcdf", "scipy")


def require_xarray(purpose: str = "NetCDF support") -> Any:
    """Import and return `xarray`, or raise `MissingDependencyError`."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            package="xarray",
            purpose=purpose,
            install_hint="pip install 'xarray>=2024.1' netCDF4",
        ) from exc
    return xr


def available_netcdf_engines() -> tuple:
    """Which NetCDF engines are actually importable in this environment."""
    found = []
    for engine in NETCDF_ENGINES:
        module = {"netcdf4": "netCDF4", "h5netcdf": "h5netcdf", "scipy": "scipy"}[engine]
        try:
            __import__(module)
        except ImportError:
            continue
        found.append(engine)
    return tuple(found)


def require_netcdf_engine(purpose: str = "reading NetCDF files") -> str:
    """Return the preferred available NetCDF engine, or raise if there is none."""
    engines = available_netcdf_engines()
    if not engines:
        raise MissingDependencyError(
            package="netCDF4",
            purpose=purpose,
            install_hint="pip install netCDF4  # or: pip install h5netcdf",
        )
    return engines[0]
