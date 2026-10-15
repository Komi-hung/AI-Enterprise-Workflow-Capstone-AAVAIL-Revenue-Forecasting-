# ============================================================
# AAVAIL Revenue Forecasting — Dockerfile
# Multi-stage build: builder → runtime
# ============================================================

# ---- Stage 1: Builder (install Python deps) ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build tools (for packages like scipy that may need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (leverages Docker cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/           ./src/
COPY models/        ./models/
COPY data/          ./data/
COPY logs/          ./logs/

# Environment defaults (override at runtime via -e)
ENV PORT=5000 \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE ${PORT}

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash aavail \
    && chown -R aavail:aavail /app
USER aavail

# Health check — poll the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# Production server: gunicorn with 4 workers
CMD ["python", "-m", "gunicorn", \
     "--workers", "4", \
     "--threads", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "src.app:app"]
