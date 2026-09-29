# NEER

**N**eural **E**mbedding based **E**stimation and **R**econstruction · SIH26066 · MoES / INCOIS

NEER is a research prototype for reconstructing subsurface ocean temperature profiles from gridded surface fields. The repository includes data loading and preprocessing, a CNN/ViT encoder with depth decoder, inference APIs, evaluation utilities, a Next.js dashboard, and a synthetic demonstration dataset.

> **Readiness:** the shipped model and dataset are for software demonstration and pipeline development. The included data are synthetic; the project has not established scientific validity against independent real-world observations. See [MODEL_CARD.md](MODEL_CARD.md) and [FINAL_AUDIT.md](FINAL_AUDIT.md).

## Run locally

Use Python 3.11+ and install the pinned dependencies from the repository root:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

The API and OpenAPI UI are at `http://localhost:8000` and `http://localhost:8000/docs`. The development configuration selects the bundled synthetic demo. For production, configure real tensor data and a compatible checkpoint; see `configs/` and the model card.

The backend allows the local frontend origins by default. Set `NEER_CORS_ORIGINS` to a comma-separated list of trusted origins when the frontend is hosted elsewhere; wildcard origins are rejected.

Run the frontend in a second terminal:

```bash
cd frontend
npm install
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local`, then run `npm run dev`. The Next.js dashboard is at `http://localhost:3000`.

Docker Compose starts both services and mounts the local `data/` and `artifacts/` directories into the backend container:

```bash
docker compose up --build
```

## Main API routes

- `/health`, `/model/info`, `/dates`
- `/reconstruct`, `/profile`, `/reconstruct/grid`, `/reconstruct/netcdf`
- `/embedding`, `/metrics`, `/evaluation/metrics`, `/evaluation/argo`
- `/explainability`, `/data/quality`, `/demo/context`

Routes are served from `backend/app/routers/`; generated values carry `data_mode` provenance. `/evaluation/argo` can demonstrate workflow behavior with generated profiles, but that is not independent observational validation.

## Model and data

- Inputs have 11 channels defined in `src/data/preprocessing/channels.py`.
- Model outputs are depth-wise predictions for normalized absolute subsurface temperature targets. Inference decodes these using training-set normalization statistics. Anomaly is derived as predicted temperature minus climatology; climatology is not added to the model output.
- The inference model is domain pooled: it produces one temperature estimate per depth for a date. Grid endpoints combine that domain-pooled temperature with the local climatology to derive spatially varying anomalies; they do not provide pixel-wise learned temperature predictions.
- The current bundled demo is synthetic (`DEMO_SYNTHETIC`). No real source data are present in `data/raw/` in the audited checkout.
- Chronological train/validation/test splits and training-only normalization are implemented. These software controls do not establish that the synthetic climatology asset has scientifically valid provenance.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.

## Development checks

```bash
python -m pytest -q
python scripts/smoke_test.py
cd frontend && npm run build
```

Additional preprocessing and evaluation tools are under `scripts/`. Evaluation reports must be interpreted together with their `data_mode` and `is_synthetic` fields; synthetic scores are not scientific performance claims.
