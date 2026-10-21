# EMOT -- CPU container: CLI + dashboard.  docker build -t emot . && docker run --rm emot --help
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dashboard]"

COPY dashboard ./dashboard
COPY results ./results
COPY docs/assets ./docs/assets

# model weights and datasets are mounted, not baked in (see docker-compose.yml)
VOLUME ["/app/weights", "/app/data", "/app/outputs"]
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s CMD curl -sf http://127.0.0.1:8050/ || exit 1

ENTRYPOINT ["emot"]
CMD ["--help"]
