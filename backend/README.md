# NEER Backend

Minimal FastAPI backend for **NEER — Neural Embedding based Estimation and Reconstruction**
(SIH26066, MoES / INCOIS).

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` — liveness check, returns project metadata.

No ML endpoints are implemented yet — this is a Phase 01 scaffold only.
