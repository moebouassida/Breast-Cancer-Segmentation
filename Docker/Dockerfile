# ─────────────────────────────────────────────────────────────────────────────
# Breast Ultrasound Segmentation — Docker Image
#
# CPU (CI / demo):
#   docker build -t breast-seg-api .
#   docker run --rm -p 8000:8000 breast-seg-api
#
# GPU (local / RunPod):
#   docker build -f Docker/Dockerfile.gpu -t breast-seg-api-gpu .
#   docker run --rm --gpus all -p 8000:8000 breast-seg-api-gpu
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/      ./src/
COPY checkpoints/ ./checkpoints/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]