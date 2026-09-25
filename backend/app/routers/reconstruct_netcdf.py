"""GET /reconstruct/netcdf — reconstructed grid as a NetCDF download (Phase 29B-2).

Thin by design: the same query params as `/reconstruct/grid` ->
`NEERRepository.reconstruct_netcdf` (real `reconstruct_grid()` output
wrapped as an `OceanDataset` and written with `save_netcdf`) ->
`FileResponse`. No grid is fabricated here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from backend.app.dependencies import get_repository
from backend.app.errors import NeerApiError
from backend.app.schemas import ErrorResponse, GridQueryParams, grid_query_params
from backend.app.services.repository import NEERRepository

router = APIRouter(tags=["reconstruct"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid parameter (e.g. out-of-domain coordinate)."},
    404: {"model": ErrorResponse, "description": "Date or grid region not available."},
    500: {"model": ErrorResponse, "description": "Inference or NetCDF generation failed."},
    503: {"model": ErrorResponse, "description": "Model, dataset, or NetCDF engine not available."},
}


def _unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


@router.get("/reconstruct/netcdf", responses=_ERROR_RESPONSES)
def reconstruct_netcdf(
    params: GridQueryParams = Depends(grid_query_params),
    repository: NEERRepository = Depends(get_repository),
) -> FileResponse:
    """Reconstructed temperature grid for one date, as a NetCDF file.

    Query parameters match `/reconstruct/grid`. The file is built from
    that reconstruction (never a placeholder dataset). Missing xarray /
    a NetCDF engine is `503`; a write/assembly failure is `500`.
    """
    try:
        path = repository.reconstruct_netcdf(
            lat_min=params.lat_min,
            lat_max=params.lat_max,
            lon_min=params.lon_min,
            lon_max=params.lon_max,
            date=str(params.date),
            depth=params.depth,
        )
    except NeerApiError:
        raise

    depth_tag = "" if params.depth is None else f"_{int(params.depth)}m"
    filename = f"neer_reconstruct_{params.date}{depth_tag}.nc"
    return FileResponse(
        path=str(path),
        media_type="application/x-netcdf",
        filename=filename,
        background=BackgroundTask(_unlink, str(path)),
    )
