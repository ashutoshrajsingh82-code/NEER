# NEER Backend API

FastAPI service for **NEER — Neural Embedding based Estimation and Reconstruction**
(SIH Problem Statement SIH26066, MoES / INCOIS).

The backend exposes real inference, evaluation, explainability, data quality, and reconstruction exports
driven by genuine checkpoints and preprocessed tensor bundles.

---

## Running Locally

```bash
# From the NEER/backend directory
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI / Swagger documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Environment Configuration

Configure the active environment via `NEER_ENVIRONMENT`:
```bash
# Options: demo, development, production (defaults to demo or config default)
export NEER_ENVIRONMENT=demo
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Architecture

Routes are strictly thin controllers:
```
HTTP Request
    │
    ▼
FastAPI Route (app/routers/*.py)
    │  Pydantic validation (app/schemas.py)
    ▼
Service Layer (app/services/repository.py & evaluation.py)
    │  Enforces domain invariants, coordinates components & LRU cache
    ├─► InferenceService (src/inference) -> PyTorch NEERModel forward pass
    ├─► Explainability (src/explainability) -> PyTorch autograd backward pass
    ├─► Evaluation (src/evaluation & src/argo_validation) -> Metrics & ARGO matching
    └─► NetCDF Engine (src/data/loaders) -> CF-compliant OceanDataset export
    ▼
HTTP Response (JSON schema or FileResponse)
```

Central exception handling (`app.exception_handler(NeerApiError)`) maps domain exceptions
to standard JSON error bodies:
```json
{
  "error": "date_not_found",
  "detail": "date 2099-01-01 is not available in the loaded dataset; call /dates for available range"
}
```

---

## Endpoints

### 1. System & Metadata
- **`GET /health`**: Service liveness and component statuses (`data`, `model`, `climatology`). Returns HTTP 200 even if degraded.
- **`GET /model/info`**: Model architecture parameters (`in_channels`, `embed_dim`, `num_depths`, `depths`, `use_gnn`, `uncertainty_enabled`), runtime device, and checkpoint metadata.
- **`GET /dates`**: List of dates available in the loaded tensor bundle with start/end bounds and data provenance (`is_synthetic`, `data_mode`).

### 2. Reconstruction & Embeddings
- **`GET /reconstruct`**: Reconstructs subsurface temperature at a specific coordinate (`lat`, `lon`, `date`, `depth`).
- **`GET /profile`**: Reconstructs the complete vertical temperature profile across all model depth levels at (`lat`, `lon`, `date`).
- **`GET /reconstruct/grid`**: Reconstructs a 2D/3D temperature field over a spatial bounding box (`lat_min`, `lat_max`, `lon_min`, `lon_max`, `date`, optional `depth`).
- **`GET /reconstruct/netcdf`**: Exports the reconstructed grid as a downloadable CF-compliant NetCDF file (`application/x-netcdf`).
- **`GET /embedding`**: Computes the domain-pooled latent representation vector (length 128) for a given date.

### 3. Explainability & Diagnostics (Phase 29B-2)
- **`GET /explainability`**: Computes gradient $\times$ input feature attribution for all 11 surface channels and generates a 2D spatial saliency map via PyTorch `autograd.grad`. Supports whole-profile or depth-specific attribution.
- **`GET /data/quality`**: Returns genuine quality metrics from the active tensor bundle: missing cell counts, valid fraction, spatial coverage, per-channel ranges, and target completeness.

### 4. Model Evaluation & Validation (Phase 29B-1)
- **`GET /metrics`**: Statistical evaluation metrics (RMSE, MAE, bias, Pearson-r, $R^2$) computed against ground-truth targets for `train`, `val`, or `test` splits.
- **`GET /evaluation/argo`**: Independent observational ARGO float profile validation report using the Phase 26 vertical interpolation and matching pipeline.

---

## Running Backend Tests

```bash
# Run all backend tests
pytest tests/test_backend_*.py -v

# Run Phase 29B-2 tests (explainability, data quality, reconstruct NetCDF)
pytest tests/test_backend_explainability.py tests/test_backend_data_quality.py tests/test_backend_reconstruct_netcdf.py -v

# Run Phase 30 API integration tests (live HTTP requests across all endpoints)
pytest tests/test_api_integration.py -v

# Run standalone live backend HTTP test runner
python scripts/test_live_backend.py
```
