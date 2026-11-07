# Stage 1: install the Python packages (prebuilt wheels only) into /deps.
FROM python:3.13-slim-trixie AS build
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: --target /deps -r requirements.txt \
 && find /deps -depth -type d -name tests -exec rm -rf {} +

# Stage 2: distroless Python 3.13 - no shell, no package manager, no pip,
# runs as the non-root user 65532. The model ships inside lid9/models.
FROM gcr.io/distroless/python3-debian13:nonroot
ENV PYTHONPATH=/deps \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    S2T_PORT=8000
COPY --from=build /deps /deps
WORKDIR /srv
COPY lid9/__init__.py ./lid9/
COPY lid9/models/lid.onnx ./lid9/models/
COPY auth.py server.py ./
EXPOSE 8000
ENTRYPOINT ["python3", "server.py"]
