# ================================================================================
# HMATS v6.8.0 - Production Runtime Dockerfile
# ================================================================================
# Build:   docker build -t hmats:6.8.0 .
# Verify:  docker run --rm --env-file .env hmats:6.8.0 --mode verify
# Paper:   docker run -d --name hmats --restart unless-stopped \
#            --env-file .env \
#            -v hmats-models:/opt/hmats/models:ro \
#            -v hmats-logs:/var/log/hmats \
#            -v hmats-data:/var/lib/hmats \
#            hmats:6.8.0 --mode paper
# ================================================================================

FROM python:3.12-slim-bookworm

LABEL maintainer="melodygaoyifan"
LABEL version="6.8.0"
LABEL description="HMATS Production Runtime (Kraken-only)"

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# HMATS paths
ENV HMATS_BASE_DIR=/opt/hmats \
    HMATS_MODEL_DIR=/opt/hmats/models \
    HMATS_LOG_DIR=/var/log/hmats \
    HMATS_STATE_DIR=/var/lib/hmats \
    HMATS_CONFIG_DIR=/opt/hmats/configs \
    HMATS_CONFIG_FILE=/opt/hmats/configs/cloud_production.json \
    HMATS_SINGLE_EXCHANGE=kraken

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 hmats \
    && useradd --uid 1000 --gid hmats --shell /bin/bash --create-home hmats

# Create directories
RUN mkdir -p /opt/hmats /var/log/hmats /var/lib/hmats \
    && chown -R hmats:hmats /opt/hmats /var/log/hmats /var/lib/hmats

WORKDIR /opt/hmats

# Install Python dependencies
COPY --chown=hmats:hmats requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt \
    && pip install --no-cache-dir ccxt filterpy ta streamlit \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sb3-contrib stable-baselines3

# Copy application code
COPY --chown=hmats:hmats main.py ./
COPY --chown=hmats:hmats core/ ./core/
COPY --chown=hmats:hmats agents/ ./agents/
COPY --chown=hmats:hmats analytics/ ./analytics/
COPY --chown=hmats:hmats configs/ ./configs/
COPY --chown=hmats:hmats defense/ ./defense/
COPY --chown=hmats:hmats drl/ ./drl/
COPY --chown=hmats:hmats engine/ ./engine/
COPY --chown=hmats:hmats exchange/ ./exchange/
COPY --chown=hmats:hmats execution/ ./execution/
COPY --chown=hmats:hmats infra/ ./infra/
COPY --chown=hmats:hmats integration/ ./integration/
COPY --chown=hmats:hmats liquidity/ ./liquidity/
COPY --chown=hmats:hmats market/ ./market/
COPY --chown=hmats:hmats orchestration/ ./orchestration/
COPY --chown=hmats:hmats portfolio/ ./portfolio/
COPY --chown=hmats:hmats risk/ ./risk/
COPY --chown=hmats:hmats signals/ ./signals/
COPY --chown=hmats:hmats strategies/ ./strategies/
COPY --chown=hmats:hmats data_mgmt/ ./data_mgmt/
COPY --chown=hmats:hmats dashboard/ ./dashboard/
COPY --chown=hmats:hmats shadow/ ./shadow/
COPY --chown=hmats:hmats tools/ ./tools/

# Volumes for persistent data
VOLUME ["/opt/hmats/models", "/var/log/hmats", "/var/lib/hmats"]

USER hmats

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os; print('OK')" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "verify"]
