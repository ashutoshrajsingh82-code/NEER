#!/usr/bin/env bash
# Run the NEER FastAPI backend locally.
set -euo pipefail
cd "$(dirname "$0")/.."
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
