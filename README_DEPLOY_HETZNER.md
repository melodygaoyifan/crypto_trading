# HMATS v6.8.0 — Hetzner Cloud Deployment

> **⚠️ PAPER-BURN-IN ERA DOC (banner added 2026-08-07).** Written for the cloud
> *paper* burn-in. The system has been **LIVE** since ~2026-04 (`--mode live
> --confirm-live --config configs/live_high_risk.json`, see
> `docker-compose.hetzner.yml`), still on **CPX21** despite the CPX31
> recommendation below, and since 2026-06-13 trades **Coinbase US perps** via the
> sleeve (Kraken structurally flat). The 72h burn-in gate and its "paper trade
> fill" criterion are historical. **Do NOT hand-build on the server** (step 10
> below is superseded): the single deploy authority is
> `bash scripts/hetzner_deploy.sh hmats` run from the operator machine (P190/P196).
> Stopping the stack with live perp positions leaves only the venue-resting
> protective stops managing them — decide about `scripts/coinbase_flatten.py`
> first.

## Capacity Verdict

**Current server: CPX21 (2 vCPU AMD EPYC, 4GB RAM, 80GB SSD, Nuremberg)**

Observed runtime on current CPX21:
- Engine: **261 MB RSS**, **1.5% CPU** at steady state
- System total: **810 MB used** (with OS + Docker overhead)
- Disk: **5.7 GB used** of 75 GB

| Scenario | CPX21 Fit | Notes |
|----------|-----------|-------|
| A. Staging / Docker bring-up | YES | Build uses ~1.5GB peak, fits in 4GB |
| B. Cloud paper burn-in | YES with swap | Engine + API + OS = ~1.5GB steady. 4GB is adequate with 2GB swap as safety net |
| C. Long-running controlled live | MARGINAL | Log/state growth over weeks + occasional memory spikes risk OOM without swap. Upgrade to CPX31 (8GB) recommended before live |

**Minimum for controlled live: CPX31 (4 vCPU, 8GB RAM) — €13.49/month**

## Deployment Recommendation

**KEEP CPX21 FOR CLOUD PAPER BURN-IN → RESCALE TO CPX31 BEFORE CONTROLLED LIVE**

CPX21 is sufficient for paper burn-in (days to weeks). Before transitioning to controlled live with real capital, rescale to CPX31 for the 2x memory headroom needed for long-running stability.

Hetzner supports in-place server resize — no data loss, just a reboot.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Hetzner CPX21/31                   │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ hmats-engine │    │  hmats-api   │               │
│  │  main.py     │    │  FastAPI     │               │
│  │  --mode paper│    │  :8080       │               │
│  │              │    │  read-only   │               │
│  └──────┬───────┘    └──────┬───────┘               │
│         │                   │                       │
│         ▼                   ▼                       │
│  ┌─────────────────────────────────────┐            │
│  │ Docker Volumes (persistent)         │            │
│  │  hmats-data: state JSONs, ledger    │            │
│  │  hmats-logs: structured logs        │            │
│  │  hmats-models: GMM/DRL (read-only)  │            │
│  └─────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

Engine writes `dashboard_state.json` and `paper_positions.json` to the shared data volume.
API reads those files (read-only mount) and serves them via HTTP.

## Exact Deployment Commands

### 1. Server Bootstrap (run once, from local machine)

```bash
# Bootstrap the server (as root)
ssh root@<IP> 'bash -s' < scripts/hetzner_bootstrap.sh
```

### 2. GitHub Deploy Key (on server)

```bash
ssh hmats   # connects as root, then:
su - hmats

# Generate deploy key
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub
# Copy the output → GitHub repo → Settings → Deploy Keys → Add (read-only)

# Configure git to use the deploy key
cat >> ~/.ssh/config << 'EOF'
Host github.com
    IdentityFile ~/.ssh/github_deploy
    StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config
```

### 3. Clone Repo

```bash
# As hmats user on server
git clone git@github.com:melodygaoyifan/crypto_trading.git ~/hmats/app
```

### 4. Place .env

```bash
# Option A: Copy from local (from your Windows machine)
scp C:\Users\melod\Downloads\hmats\.env hmats:/home/hmats/hmats/app/.env

# Option B: Create on server from template
cp ~/hmats/app/env/.env.template ~/hmats/app/.env
nano ~/hmats/app/.env   # fill in API keys

# Lock permissions
chmod 600 ~/hmats/app/.env
```

### 5. Upload Models

From local Windows:
```powershell
scp -r C:\Users\melod\Downloads\hmats\models\regime_classifier hmats:/home/hmats/hmats/models/
scp -r C:\Users\melod\Downloads\hmats\models\decision_transformer hmats:/home/hmats/hmats/models/
# If DRL models are trained:
# scp -r C:\Users\melod\Downloads\hmats\models\retrained hmats:/home/hmats/hmats/models/
```

### 6. Build and Start

```bash
# As hmats user on server
cd ~/hmats/app

# Build images
docker compose -f docker-compose.hetzner.yml build

# Sync models to Docker volume
docker volume create hmats-models 2>/dev/null || true
docker run --rm \
  -v hmats-models:/models \
  -v /home/hmats/hmats/models:/src:ro \
  alpine sh -c "cp -r /src/* /models/"

# Start services
docker compose -f docker-compose.hetzner.yml up -d

# Verify
docker ps
docker logs hmats-engine --tail 20
curl -s localhost:8080/health | python3 -m json.tool
```

### 7. Validate

From local:
```bash
bash scripts/hetzner_validate.sh hmats
```

### 8. Verify Mode (one-shot test)

```bash
# On server
docker compose -f docker-compose.hetzner.yml run --rm hmats-engine --mode verify
```

### 9. Log Inspection

```bash
# Engine logs (live)
docker logs -f hmats-engine

# API logs
docker logs -f hmats-api

# Recent proof logs
docker exec hmats-engine ls /opt/hmats/logs/

# API status check
curl -s localhost:8080/status | python3 -m json.tool
curl -s localhost:8080/positions/current | python3 -m json.tool
curl -s localhost:8080/pnl/summary | python3 -m json.tool
curl -s localhost:8080/regime/current | python3 -m json.tool
```

### 10. Stop / Restart

```bash
# Stop all
docker compose -f docker-compose.hetzner.yml down

# Restart engine only
docker compose -f docker-compose.hetzner.yml restart hmats-engine

# Full rebuild after code update
# [SUPERSEDED 2026-08-07 — do not hand-build on the server (P190).
#  Deploy from the operator machine instead:]
#   bash scripts/hetzner_deploy.sh hmats
```

## API Endpoints

All read-only, bound to `127.0.0.1:8080` (not exposed externally).

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness + engine state freshness |
| `GET /status` | System mode, equity, PnL, tick count |
| `GET /signals/latest` | Per-asset quant signals, regime, alpha gate |
| `GET /regime/current` | Regime classification per asset |
| `GET /positions/current` | Open positions with entry price, direction |
| `GET /pnl/summary` | Equity, drawdown, win rate, trade count |
| `GET /logs/recent?lines=50` | Recent engine log lines (max 200) |
| `GET /drl/status` | DRL gate level, inference status |
| `GET /sentiment/latest` | Sentiment L1 scores per asset |
| `GET /docs` | Swagger UI |

## Burn-In Decision Gate

### A. Cloud Burn-In PASS Criteria (paper on CPX21)

All of these must be true for ≥72 hours continuous:

1. **Engine uptime**: No container restart (check `docker inspect hmats-engine --format '{{.RestartCount}}'` = 0)
2. **Tick continuity**: `dashboard_state.json` updated_at never older than 5 hours (4H tick + 1H tolerance)
3. **Memory stable**: Engine RSS stays below 1.5 GB (check `docker stats --no-stream`)
4. **No CRITICAL logs**: `docker logs hmats-engine 2>&1 | grep -c CRITICAL` = 0
5. **API responsive**: `/health` returns `healthy` on every check
6. **Disk growth**: Less than 500 MB/week log+state growth
7. **Trades execute**: At least 1 paper trade fill recorded in shadow ledger
8. **No OOM kills**: `dmesg | grep -i oom` returns nothing

### B. Mandatory Rescale Triggers (before controlled live)

Rescale to CPX31 (8GB) if ANY of:

1. Engine RSS exceeds 2.5 GB at any point during burn-in
2. Swap usage exceeds 500 MB sustained (>1 hour)
3. Container restarts due to OOM
4. Docker build fails due to insufficient memory
5. Disk usage exceeds 60% (need headroom for log rotation)
6. CPU usage sustained above 80% during tick processing

### Rescale Procedure (Hetzner)

1. Hetzner Console → Server → Rescale → Select CPX31
2. Server reboots (1-2 minutes downtime)
3. Dead-man switch will fire (Kraken cancels open orders after 60s — this is safe)
4. After reboot: `docker compose -f docker-compose.hetzner.yml up -d`
5. Validate: `bash scripts/hetzner_validate.sh hmats`

## Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| CPX21 OOM during prolonged paper run | MEDIUM | 2GB swap configured in bootstrap; monitor with `docker stats` |
| Engine crash loses in-flight tick state | LOW | Fail-closed design; next tick starts clean. Paper positions persisted atomically |
| Kraken API rate limit on cold start | LOW | Built-in rate limiter (15 req/s). First tick fetches more data than steady state |
| Model files out of sync with code | MEDIUM | Models in read-only volume. Must manually scp after retraining |
| Log disk growth over weeks | LOW | Logrotate configured for 14-day retention. Docker json-file driver capped at 50MB × 5 |
| Network partition to Kraken | LOW | Dead-man switch cancels all orders after 60s timeout. Reconnect handler in engine |
| dashboard_state.json write contention | NEGLIGIBLE | Atomic tmp+rename write pattern. API reads with 5s cache |
