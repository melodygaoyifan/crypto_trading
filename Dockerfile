# ================================================================================
# HMATS v5.1.0 - Production Runtime Dockerfile
# ================================================================================
# Purpose: Runtime-only container (no training dependencies)
# Target: Single-exchange (Kraken), cost-optimized cloud deployment
#
# Build:
#   docker build -t hmats:5.1.0 .
#
# Run (paper trading):
#   docker run -d --name hmats \
#     -e KRAKEN_API_KEY=xxx \
#     -e KRAKEN_API_SECRET=xxx \
#     -v /path/to/models:/opt/hmats/models:ro \
#     -v /var/log/hmats:/var/log/hmats \
#     -v /var/lib/hmats:/var/lib/hmats \
#     hmats:5.1.0 --mode paper
#
# ================================================================================

# Use slim Python image for minimal footprint
FROM python:3.11-slim-bookworm AS base

# Labels
LABEL maintainer="HMATS Team"
LABEL version="5.1.0"
LABEL description="HMATS Production Runtime (Single-Exchange, Kraken-only)"

# =============================================================================
# BUILD ARGUMENTS
# =============================================================================

ARG HMATS_VERSION=5.1.0
ARG PYTHON_VERSION=3.11

# =============================================================================
# ENVIRONMENT
# =============================================================================

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# HMATS settings
ENV HMATS_VERSION=${HMATS_VERSION} \
    HMATS_BASE_DIR=/opt/hmats \
    HMATS_MODEL_DIR=/opt/hmats/models/current \
    HMATS_LOG_DIR=/var/log/hmats \
    HMATS_STATE_DIR=/var/lib/hmats \
    HMATS_CONFIG_DIR=/opt/hmats/configs \
    HMATS_CONFIG_FILE=/opt/hmats/configs/cloud_production.json

# Single exchange mode
ENV HMATS_SINGLE_EXCHANGE=kraken

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    curl \
    ca-certificates \
    # Build dependencies (for some Python packages)
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# CREATE USER (non-root)
# =============================================================================

RUN groupadd --gid 1000 hmats \
    && useradd --uid 1000 --gid hmats --shell /bin/bash --create-home hmats

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

RUN mkdir -p \
    /opt/hmats \
    /opt/hmats/configs \
    /opt/hmats/models \
    /var/log/hmats \
    /var/lib/hmats \
    && chown -R hmats:hmats /opt/hmats /var/log/hmats /var/lib/hmats

# =============================================================================
# PYTHON DEPENDENCIES (runtime only, no training)
# =============================================================================

WORKDIR /opt/hmats

# Copy requirements first for better caching
COPY --chown=hmats:hmats requirements-runtime.txt ./

# Install runtime dependencies
RUN pip install --no-cache-dir -r requirements-runtime.txt

# =============================================================================
# COPY APPLICATION CODE
# =============================================================================

# Copy only runtime-necessary files (exclude training, tests, etc.)
COPY --chown=hmats:hmats main.py main_v36.py ./
COPY --chown=hmats:hmats core/ ./core/
COPY --chown=hmats:hmats agents/ ./agents/
COPY --chown=hmats:hmats analytics/ ./analytics/
COPY --chown=hmats:hmats configs/ ./configs/
COPY --chown=hmats:hmats defense/ ./defense/
COPY --chown=hmats:hmats drl/ ./drl/
COPY --chown=hmats:hmats engine/ ./engine/
COPY --chown=hmats:hmats exchange/kraken/ ./exchange/kraken/
COPY --chown=hmats:hmats exchange/__init__.py ./exchange/
COPY --chown=hmats:hmats execution/ ./execution/
COPY --chown=hmats:hmats infra/ ./infra/
COPY --chown=hmats:hmats liquidity/ ./liquidity/
COPY --chown=hmats:hmats market/ ./market/
COPY --chown=hmats:hmats market_analysis/ ./market_analysis/
COPY --chown=hmats:hmats orchestration/ ./orchestration/
COPY --chown=hmats:hmats overrides/ ./overrides/
COPY --chown=hmats:hmats risk/ ./risk/
COPY --chown=hmats:hmats signals/ ./signals/
COPY --chown=hmats:hmats integration/ ./integration/
COPY --chown=hmats:hmats data_mgmt/ ./data_mgmt/

# Copy legacy (frozen, for compatibility)
COPY --chown=hmats:hmats legacy/ ./legacy/

# Copy model configs (not weights - those are mounted)
COPY --chown=hmats:hmats models/decision_transformer/config.json ./models/decision_transformer/
COPY --chown=hmats:hmats models/regime_classifier/gmm_config.json ./models/regime_classifier/
COPY --chown=hmats:hmats models/llm_cache/model_config.json ./models/llm_cache/

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

# Switch to non-root user
USER hmats

# Volumes for persistent data
VOLUME ["/opt/hmats/models", "/var/log/hmats", "/var/lib/hmats"]

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from core.cloud_config import get_cloud_config; c = get_cloud_config(); print('OK')" || exit 1

# =============================================================================
# ENTRYPOINT
# =============================================================================

# Default command: verify mode (safe)
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "verify"]

# Example commands:
# Paper trading:  docker run hmats:5.1.0 --mode paper
# Live (DANGER):  docker run hmats:5.1.0 --mode live --confirm-live
