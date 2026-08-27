# Server crons — the standing list (P418, 2026-08-27)

Hetzner box `hmats` (ssh alias). Two crontabs exist. Nothing here trades:
every job is a read-only evidence tool, an accumulator, or an alarm. This
file is the record; the crontabs are the truth — read them with
`ssh hmats "crontab -u hmats -l; crontab -l"` before changing anything.

## The rule that matters

**Nothing may `docker cp` into the live container, edit files inside it, or
edit the checkout at `/home/hmats/hmats/app` by hand.** The engine runs the
image built from a CI-green commit of `origin/main` (`scripts/hetzner_deploy.sh`
refuses to build anything else — after its pull it asserts the server's HEAD
equals the CI-verified sha AND the server tree is clean, P418). A file copied
or edited on the server exists in no commit, survives until the next
`--force-recreate` and then silently disappears, and makes the running code
differ from what every gate verified. If a script the crons need is missing
from the image, add it to the `Dockerfile.engine` allowlist AND the
`.dockerignore` negation (both halves, P192) and deploy.

The one historical exception is `seat_check_weekly.sh`, a host-side wrapper
that resolves the newest report at run time (P295c); it lives on the host, not
in the container.

## `hmats` user — the weekly evidence pipeline (Mondays, UTC)

All `docker exec hmats-engine ...` unless noted; stdout appends to
`/home/hmats/evidence_cron.log`; reports land on the `hmats-data` volume
(`/opt/hmats/data/evidence_reports`), which `scripts/september_check.py` pulls.

| time | job | what it is |
|---|---|---|
| 06:10 | `analytics/ic/agent_ic_review.py --window-days 30 --report-dir /opt/hmats/data/evidence_reports` | per-agent forward IC through the P166 bar (P230) |
| 06:15 | `analytics/ic/agent_disagreement_review.py --advisor whale --window-days 90` | whale entry-filter counterfactual (P324) |
| 06:16 | `analytics/ic/agent_disagreement_review.py --advisor model_alpha --window-days 90` | model_alpha entry-filter counterfactual (P324) |
| 06:20 | `analytics/calibration/slope_calibrator.py --window-days 90 --report-dir /opt/hmats/data/evidence_reports` | honest alpha-slope calibrator (P232) |
| 06:25 | `analytics/calibration/tripwire_check.py` | P237 streak DETECTION only — prescription retired (P299); it never edits config |
| 06:30 | `analytics/sleeve_attribution/sleeve_beta_review.py --window-days 60` | realized sleeve beta (P230) |
| 06:35 | `scripts/trend_regime_review.py` | trend-gate forward evidence (P198/P213) |
| 06:40 | `sh /home/hmats/seat_check_weekly.sh` (host wrapper) | the P295 seat controller on the newest IC report; exit 3 = recommendation, never a config edit |
| 06:45 | `scripts/calibration_check.py` | measured-constant staleness (P327) |
| 06:50 | `scripts/september_check.py --countdown-only` | the September read-date alarm (P333); exit 3 = a read window has closed |
| 06:55 | host venv: `training/scripts/fetch_coinglass_history.py --interval 4h` (`HMATS_COINGLASS_DIR=/home/hmats/coinglass_history`) | derivatives-positioning accumulation, MERGE-not-overwrite (P389c) |
| 06:57 | host venv: `training/scripts/fetch_coinglass_history.py --interval 1d` | same, daily interval |

The two CoinGlass jobs run in `/home/hmats/cg_venv` (the runtime image has
pandas but not pyarrow — P389c); they write to a persistent host directory,
never into the container.

## `root` — the three remaining jobs

After the 2026-08-27 cleanup exactly these remain. They run in the host
checkout via `cd /home/hmats/hmats/app` and read either a persistent host
snapshot dir or the `hmats-data` volume; none writes into the container.

| time | job | what it is |
|---|---|---|
| 07:10 daily | `training/scripts/accumulate_newdata_snapshots.py` (`HMATS_SNAPSHOT_DIR=/home/hmats/signal_snapshots`, cg_venv) | banks the options put/call + on-chain daily rows the free feeds serve as snapshots only (P395) |
| 07:20 Mon | `training/scripts/newdata_gated_probe.py` (same env) | the gated auto-probe: prints `accumulating N/180` until ~180 days exist, then runs the hold-aware Rung-0 (P396) |
| 07:40 Mon | `/usr/bin/python3 training/scripts/etfflow_timing_check.py --ledger-dir /var/lib/docker/volumes/hmats-data/_data/strategy_shadow` | the ETF-flow leak/timing check against the live ledger (P404b): LEAK exit 3 = do NOT arm |

The root crontab header comment (`[TREND-SHADOW 2026-06-14]`) predates the
current jobs and describes nothing that still runs; the three jobs above are
the whole list.

## Changing a cron

Install crontabs by `scp` + `crontab <file>` (or `crontab -u hmats <file>`),
never through a nested-quoted heredoc over ssh — a `$HOME` in the heredoc
expanded on the LOCAL shell once and wrote the operator's Windows home path
into the server crontab (P235). After any change, re-read the crontab back
and update the tables here in the same commit (Rule 7).
