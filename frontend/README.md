# NEER Frontend

Minimal Next.js (App Router) frontend for **NEER — Neural Embedding based
Estimation and Reconstruction** (SIH26066, MoES / INCOIS).

## Run locally

Start the backend from the repository root first:

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Set the frontend API origin in `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The repository includes a local ignored `.env.local` for this checkout. If it is missing, copy `.env.example` to `.env.local`. Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000. The frontend reads `NEXT_PUBLIC_API_URL` when Next.js starts; restart the dev server after changing `.env.local`. The backend should return `DEMO_SYNTHETIC` from `/health` when running with the default demo configuration.
