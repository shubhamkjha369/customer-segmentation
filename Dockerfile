# ============================================================
# Customer Segmentation — Production Dockerfile
# ============================================================
# Multi-stage build for a lean production image
# Base: Python 3.11-slim
# Exposes port 8501 (Streamlit)
# ============================================================

# ── Stage 1: dependency builder ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for pandas / scipy / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies into a prefix (for copying)
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: production runner ───────────────────────────────
FROM python:3.11-slim AS runner

LABEL maintainer="Your Name <you@example.com>"
LABEL description="Customer Segmentation ML System"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# System runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY config.yaml            .
COPY requirements.txt       .
COPY src/                   ./src/
COPY app/                   ./app/
COPY models/                ./models/
COPY data/processed/        ./data/processed/

# ── Streamlit config ─────────────────────────────────────────
RUN mkdir -p /app/.streamlit
COPY docker/streamlit_config.toml /app/.streamlit/config.toml 2>/dev/null || true

RUN printf '[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
[theme]\n\
base = "dark"\n\
primaryColor = "#00d4ff"\n\
' > /app/.streamlit/config.toml

# Non-root user for security
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Environment
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
