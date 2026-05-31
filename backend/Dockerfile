FROM python:3.12-slim

# curl is needed for Docker health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BM25 sparse embedding model so first request doesn't stall
RUN python -c "from fastembed import SparseTextEmbedding; m=SparseTextEmbedding(model_name='Qdrant/bm25'); list(m.embed(['warmup'])); print('BM25 model ready')"

COPY app ./app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Prepare data directory layout
RUN mkdir -p /app/data/uploads /app/data/chunks /app/data/markdown /app/data/reports

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
