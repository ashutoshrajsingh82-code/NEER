# NEER Phase 50 — Final Audit

**Audit date:** 2026-09-29  
**Repository:** NEER project checkout  
**Purpose:** final engineering and scientific-readiness audit of the checked-in project state.

## Executive result

| Area | Status | Finding |
|---|---|---|
| Backend and Python test suite | **PASS** | Final rerun: 1,416 passed, 2 skipped. |
| Backend demo smoke workflow | **PASS** | Model load, inference, core API responses, NetCDF round trip, and frontend API configuration passed. |
| Frontend production build | **PASS** | Next.js build completed; font download and two lint warnings remain. |
| Frontend unit tests | **PASS** | Vitest now transforms JSX in `.js` components correctly; 315 tests passed across 16 files. |
| Local demo startup | **PASS** | `frontend/.env.local` points to the backend, Next.js loads it, and `/demo/context` returns a demo selection in development. |
| Real-data operation | **NOT VERIFIED** | No real source dataset is present in `data/raw/`; the shipped data and checkpoint are synthetic-demo assets. |
| Independent ARGO validation | **NOT AVAILABLE** | No real ARGO profiles are present. Demo ARGO output is generated and is not observational validation. |
| Scientific readiness | **NOT ESTABLISHED** | Synthetic evidence only; the trained checkpoint underperformed the climatology and ridge baselines on the synthetic test split. |

Engineering paths are functional for a labeled synthetic demonstration. The evidence does not support oceanographic conclusions or operational use.

## Critical finding fixed

The trained model target is normalized absolute `subsurface_temp`; preprocessing does not subtract climatology before training. Inference nevertheless treated the de-normalized model output as an anomaly and added climatology to it. This could produce an invalid reported temperature. For the exercised demo case (2021-12-01, 15.5°N, 72.75°E, 100 m), the old output was about 40.46°C. The model's de-normalized temperature was 17.66°C and climatology was about 22.80°C; the corrected anomaly is therefore about −5.15°C.

Inference now keeps the model temperature in physical units, reports anomaly as `temperature - climatology` when climatology is available, and still returns model temperature when climatology is missing. Profile, grid, explainability, and model helper semantics were aligned. The model is domain-pooled: grid responses repeat the model's one temperature estimate per depth; spatial anomaly variation comes from local climatology, not pixel-wise learned temperature predictions.

## Tests and checks run

- Python suite: `.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp=.pytest-phase-user-fix -p no:cacheprovider` — **1,416 passed, 2 skipped, 61 warnings** in 146.92 seconds. Warnings are dependency deprecations/runtime notices and a Pydantic protected-namespace warning; no test failed.
- Smoke test: `.venv\Scripts\python.exe scripts/smoke_test.py` — **8/8 checks passed**, including demo data provenance, checkpoint load, 256D embedding, 15 depth levels, inference, core endpoints, NetCDF export/reopen, and frontend API URL configuration.
- Frontend build: `npm run build` — **passed**, all 16 static pages generated. It could not download Google Fonts in this environment; lint reported the existing custom-font placement and anonymous default-export warnings.
- Frontend unit tests: `npm test -- --run` — **315 passed across 16 files**. Fixed JSX-in-`.js` transformation in Vitest, aligned dates/model-info validation contracts, corrected the ARGO route and NetCDF filename/header handling, and updated explainability validation to match absolute-temperature predictions.
- Local demo startup: created ignored `frontend/.env.local` with `NEXT_PUBLIC_API_URL=http://localhost:8000`, restarted Next.js, and verified `GET /dashboard` returns HTTP 200 and `GET /demo/context` returns a `DEMO_SYNTHETIC` context in the development environment.
- Preprocessing dry run: `python scripts/preprocess_data.py --environment demo --dry-run --quiet` — passed in the preceding audit.
- Evaluation CLI: `scripts/run_evaluation.py` ran against the shipped tensor bundle and checkpoint. It labeled the report `DEMO_SYNTHETIC` / `is_synthetic=true` and wrote the report and plots to a temporary audit directory. On the synthetic test split, normalized-target RMSE was **1.1659** for NEER, **0.2466** for climatology, and **0.2924** for ridge. The test split contains four monthly samples (60 masked profile observations in the report). These are software-pipeline results, not scientific performance estimates.
- API and file workflows: the Python integration suites cover health, reconstruction, profile/grid, embeddings, metrics, explainability, data quality, evaluation, ARGO demo behavior, and NetCDF responses. The smoke test independently exercised the core live application test client path.
- `git diff --check` reported an existing extra blank line at EOF in `frontend/components/ocean-map/mapLegendControls.test.js`; Phase 50 did not change it. Line-ending notices are also reported by Git on this Windows checkout.

## Data, model, and validation evidence

- `data/demo/neer_demo_ocean_dataset.npz` and `data/processed/neer_tensors.npz` are labeled synthetic. The demo has 24 monthly time steps, 15 depth levels, and a 101 × 241 grid; the chronological tensor split is 16 train, 4 validation, and 4 test steps.
- Training metadata records normalization fitted on the training window and a checkpoint/history from a synthetic-demo run. This verifies the software training path, not observational model skill.
- `data/raw/` contains only `.gitkeep`. No real oceanographic source files or ARGO observations were available in the audited checkout.
- A demo ARGO workflow exists and is tested, but it produces generated profiles. It must not be called independent observational validation. Real ARGO evaluation cannot be run until traceable profiles and matchup inputs are supplied.
- The climatology asset exists, but the audit could not establish its independent source and reference-period provenance. This is a scientific-data warning even though the code path works.
- The optional GNN and uncertainty output are disabled by default. The uncertainty implementation does not claim calibration.

## Application and deployment notes

- The FastAPI application and tested demonstration routes load the model and demo data successfully. `/evaluation/argo` demo behavior is available; an observational ARGO result is not.
- Docker Compose previously used a frontend environment variable the API client does not read and did not mount the model/data directories. It now sets `NEXT_PUBLIC_API_URL` and mounts `data/` and `artifacts/` into the backend container.
- `backend/app/main.py` now allows explicit local frontend origins by default and accepts a comma-separated `NEER_CORS_ORIGINS` deployment allowlist; wildcard origins are rejected. `Content-Disposition` is exposed for browser NetCDF downloads.
- Frontend build success verifies compilation and static generation, not visual behavior across browsers or viewport sizes; that manual visual audit was not completed.

## Phase 50 files changed

- `src/inference/service.py` — corrected temperature/anomaly interpretation and no-climatology behavior.
- `src/models/neer_model.py`, `src/models/__init__.py`, `src/inference/__init__.py` — aligned model and inference documentation/helper semantics with the absolute-temperature training target.
- `backend/app/services/repository.py` — aligned explainability target validation and labeling.
- `backend/app/main.py` — replaced wildcard credentialed CORS with an explicit configurable origin allowlist.
- `tests/test_cors.py` — checks the explicit CORS allowlist and preflight behavior.
- `frontend/lib/api.js`, `frontend/lib/api.test.js` — corrected API validation, ARGO route, and NetCDF download filename/header handling.
- `frontend/lib/explainabilityResponse.js`, `frontend/lib/explainabilityResponse.test.js` — corrected explainability validation for absolute temperature output and missing climatology.
- `frontend/lib/embeddingExplorer.test.js` — corrected the invalid-dimension fixture.
- `frontend/vitest.config.js` — enabled JSX transformation in component `.js` modules.
- `frontend/.env.local`, `frontend/README.md` — configured the local frontend API origin and corrected current startup instructions.
- `backend/app/services/repository.py` — allow demo context in development when the demo data flag is enabled.
- `src/inference/test_inference_service.py`, `tests/test_neer_model.py`, `tests/test_gnn.py`, `tests/test_backend_reconstruct.py`, `tests/test_backend_reconstruct_grid.py`, `tests/test_backend_explainability.py`, `tests/test_api_integration.py` — changed assertions to enforce the corrected data contract.
- `docker-compose.yml` — corrected API URL configuration and mounted required data/checkpoint directories.
- `README.md`, `MODEL_CARD.md` — replaced stale phase claims with current model, data, setup, and limitations.
- `FINAL_AUDIT.md` — recorded this audit and its evidence.

## Final determination

**Engineering status: PASS WITH WARNINGS for a synthetic demonstration.** Python coverage, smoke workflow, frontend tests/build, inference, API integration, and NetCDF file operations pass. The frontend was not visually audited manually.

**Scientific status: NOT VALIDATED.** The current checked-in demonstration is synthetic, real ARGO data are absent, climatology provenance is incomplete, and synthetic test-split metrics show NEER underperforming both evaluated baselines. No scientific or operational readiness claim is warranted.
