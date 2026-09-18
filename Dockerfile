# NEER — Backend Dockerfile
# Neural Embedding based Estimation and Reconstruction (SIH26066, MoES/INCOIS)
#
# Builds the FastAPI backend. The frontend has its own Dockerfile
# (frontend/Dockerfile), orchestrated together via docker-compose.yml.

FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal for this scaffold phase.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY src/ src/
COPY configs/ configs/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
