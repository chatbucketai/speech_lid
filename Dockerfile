# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt -r requirements-api.txt

COPY python/ ./python/

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/python/lid9/ ./lid9/
COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
