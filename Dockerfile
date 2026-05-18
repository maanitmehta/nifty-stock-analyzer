FROM python:3.9-slim

WORKDIR /app

# Install OS-level build tools needed by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first so Docker can cache this layer
COPY pyproject.toml .

# Install the package and all runtime dependencies (no dev extras)
RUN pip install --no-cache-dir .

# Copy application source and static universe data
COPY src/  src/
COPY data/universe/  data/universe/
COPY scripts/  scripts/

# Pre-create cache directory (will be populated at runtime)
RUN mkdir -p data/cache

# Streamlit runs on 8501 by default
EXPOSE 8501

# Liveness check — Streamlit exposes a health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run as non-root for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["streamlit", "run", "src/nifty_analyzer/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
