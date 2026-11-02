# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY python/ ./python/

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/python/lid9/ ./lid9/

COPY requirements.txt .

EXPOSE 8000

ENTRYPOINT ["python", "-m", "lid9.cli"]
