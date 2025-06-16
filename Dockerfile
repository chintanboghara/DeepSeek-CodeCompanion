# 1. Base Image (pinned to a specific patch version)
FROM python:3.10.12-slim AS base

# 2. Environment Variables (combined for brevity)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# 3. Create a non‑root user and set working directory
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /home/appuser/app \
    && chown appuser:appuser /home/appuser/app
WORKDIR /home/appuser/app

# 4. Install system‑level dependencies (e.g., for healthchecks or future native libs)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies as root (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy application code and give ownership to non‑root user
COPY --chown=appuser:appuser . .

# 7. Switch to non‑root for runtime
USER appuser

# 8. Expose Streamlit port
EXPOSE 8501

# 9. Healthcheck to ensure the app is up
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD curl --fail http://localhost:8501/ || exit 1

# 10. Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]