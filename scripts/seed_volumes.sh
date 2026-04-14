#!/usr/bin/env bash
# Seed Hetzner Docker volumes with local models + critical state.
# Run this ON the Hetzner host AFTER `rsync`-ing models/ and data/ into
# ~/hmats/seed/{models,data} from the laptop.
#
# Usage (on Hetzner):
#   rsync -avz --progress laptop:/c/Users/melod/Downloads/hmats/models/ ~/hmats/seed/models/
#   rsync -avz --progress laptop:/c/Users/melod/Downloads/hmats/data/drl_promotion_state.json \
#     laptop:/c/Users/melod/Downloads/hmats/data/tranche_state.json \
#     ~/hmats/seed/data/
#   bash scripts/seed_volumes.sh

set -euo pipefail

SEED_DIR="${SEED_DIR:-$HOME/hmats/seed}"
MODELS_SRC="$SEED_DIR/models"
DATA_SRC="$SEED_DIR/data"

if [[ ! -d "$MODELS_SRC" ]]; then
    echo "[seed] ERROR: $MODELS_SRC missing. rsync models/ from laptop first." >&2
    exit 1
fi

echo "[seed] Creating named volumes if absent..."
docker volume create hmats-models >/dev/null
docker volume create hmats-data >/dev/null
docker volume create hmats-logs >/dev/null

echo "[seed] Copying models ($(du -sh "$MODELS_SRC" | cut -f1)) into hmats-models..."
docker run --rm \
    -v hmats-models:/dest \
    -v "$MODELS_SRC":/src:ro \
    alpine sh -c "cp -r /src/. /dest/ && chown -R 1000:1000 /dest"

if [[ -d "$DATA_SRC" ]]; then
    echo "[seed] Copying critical state files into hmats-data..."
    docker run --rm \
        -v hmats-data:/dest \
        -v "$DATA_SRC":/src:ro \
        alpine sh -c "cp -r /src/. /dest/ && chown -R 1000:1000 /dest"
else
    echo "[seed] No $DATA_SRC — hmats-data will start empty (engine re-creates)."
fi

echo "[seed] Verifying volume contents..."
docker run --rm -v hmats-models:/v:ro alpine sh -c "ls /v | head -20 && echo --- && du -sh /v"
docker run --rm -v hmats-data:/v:ro alpine sh -c "ls /v | head -20"

echo "[seed] Done. Volumes ready for: docker compose -f docker-compose.hetzner.yml up -d"
