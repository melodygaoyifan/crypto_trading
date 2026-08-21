#!/usr/bin/env bash
# scripts/hmats_monitor.sh — one-shot health snapshot for the live Hetzner engine.
#
# Reports on 8 axes, each with a clear PASS/WARN/FAIL marker:
#   1. Container state (engine + api + dashboard)
#   2. Engine healthcheck status
#   3. Recent restarts / fatal errors
#   4. Tick liveness (last LIVE_DATA / AGENT-TRACE)
#   5. Fusion behavior (DECIDE_CONSENSUS vs DECIDE_CONFLICT vs DECIDE_SOLO)
#   6. Exit-DRL state (per-asset modes + kill-switch trips)
#   7. Trades + alpha-gate activity
#   8. Resource usage
#
# Usage:
#   bash scripts/hmats_monitor.sh           # one-shot
#   bash scripts/hmats_monitor.sh --watch   # loop every 60s (Ctrl-C to stop)
#
# Exit code is non-zero if any FAIL is observed (suitable for cron).

# [P363] pipefail: without it a pipeline reports its LAST stage's status,
# so `cmd | grep` reads grep's exit code and a failing cmd looks fine
# (P185's trap). Nothing here branches on $? after a pipeline today —
# it captures output and line ~40 already appends `|| true` — so this
# is behaviour-preserving; it closes the class for the next line added.
# Only -o pipefail, deliberately: -e would abort this monitor on the
# first probe that legitimately returns non-zero.
set -u
set -o pipefail

HOST="${HMATS_HOST:-hmats}"
WATCH=0
[[ "${1:-}" == "--watch" ]] && WATCH=1

CYAN=$'\e[36m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; DIM=$'\e[2m'; RESET=$'\e[0m'
PASS="${GREEN}OK${RESET}"
WARN="${YELLOW}WARN${RESET}"
FAIL="${RED}FAIL${RESET}"

# ssh_count <pattern> <since> — return a single integer count of grep -c matches
# Strips all non-digit characters; defaults to 0 on any error/empty.
ssh_count() {
  local pattern="$1" since="$2"
  # `grep -c` always prints a number (0 when no matches) and exits 1.
  # We don't need `|| echo 0` — that would double the output.
  # `|| true` keeps the SSH command exit code from propagating back.
  local raw
  raw=$(ssh -o ConnectTimeout=10 "$HOST" \
    "docker logs hmats-engine --since $since 2>&1 | grep -cE '$pattern' || true" \
    2>/dev/null)
  # Strip everything except first integer
  local n
  n=$(printf '%s' "$raw" | tr -dc '0-9' | head -c 8)
  echo "${n:-0}"
}

snapshot() {
  local fail_count=0
  local now
  now=$(date -u +"%Y-%m-%d %H:%M:%SZ")

  echo "${CYAN}════════════════════════════════════════════════════════════════════${RESET}"
  echo "  HMATS engine snapshot — host=${HOST} — ${now}"
  echo "${CYAN}════════════════════════════════════════════════════════════════════${RESET}"

  # 1. Container state ------------------------------------------------
  echo ""
  echo "${CYAN}[1] CONTAINER STATE${RESET}"
  local ps_out
  ps_out=$(ssh -o ConnectTimeout=10 "$HOST" \
    "docker ps --format '{{.Names}}|{{.Status}}|{{.RunningFor}}' | grep hmats" 2>/dev/null)
  if [[ -z "$ps_out" ]]; then
    echo "  $FAIL  ssh/docker unreachable"
    fail_count=$((fail_count + 1))
  else
    while IFS='|' read -r name status running_for; do
      if [[ "$status" == *unhealthy* ]]; then
        echo "  $FAIL  $name — $status"
        fail_count=$((fail_count + 1))
      elif [[ "$status" == *healthy* || "$status" == *Up* ]]; then
        echo "  $PASS  $name — $status"
      else
        echo "  $FAIL  $name — $status"
        fail_count=$((fail_count + 1))
      fi
    done <<< "$ps_out"
  fi

  # 2. Engine healthcheck ---------------------------------------------
  echo ""
  echo "${CYAN}[2] ENGINE HEALTHCHECK${RESET}"
  local hc
  hc=$(ssh "$HOST" "docker inspect hmats-engine --format \
    'health={{.State.Health.Status}}|streak={{.State.Health.FailingStreak}}|restarts={{.RestartCount}}'" \
    2>/dev/null)
  if [[ -z "$hc" ]]; then
    echo "  $FAIL  inspect unavailable"
    fail_count=$((fail_count + 1))
  else
    local hstat hstreak hrestarts
    hstat=$(printf '%s' "$hc" | sed -n 's/.*health=\([^|]*\).*/\1/p')
    hstreak=$(printf '%s' "$hc" | sed -n 's/.*streak=\([0-9]*\).*/\1/p')
    hrestarts=$(printf '%s' "$hc" | sed -n 's/.*restarts=\([0-9]*\).*/\1/p')
    if [[ "$hstat" == "healthy" ]]; then
      echo "  $PASS  health=$hstat  streak=${hstreak:-0}  restarts=${hrestarts:-0}"
    elif [[ "$hstat" == "starting" ]]; then
      echo "  $WARN  health=$hstat (start_period grace window)  restarts=${hrestarts:-0}"
    else
      echo "  $FAIL  health=$hstat  streak=${hstreak:-0}  restarts=${hrestarts:-0}"
      fail_count=$((fail_count + 1))
    fi
  fi

  # 3. Restarts + recent fatal errors ---------------------------------
  echo ""
  echo "${CYAN}[3] RECENT FATAL ERRORS (last 30m)${RESET}"
  local fatal_count
  fatal_count=$(ssh_count 'CRITICAL|FATAL|Traceback' '30m')
  if [[ "$fatal_count" -eq 0 ]]; then
    echo "  $PASS  0 critical/fatal/traceback lines in last 30 min"
  elif [[ "$fatal_count" -lt 5 ]]; then
    echo "  $WARN  $fatal_count critical-level entries"
    ssh "$HOST" "docker logs hmats-engine --since 30m 2>&1 | grep -E 'CRITICAL|FATAL|Traceback' | tail -3" \
      2>/dev/null | sed 's/^/        /'
  else
    echo "  $FAIL  $fatal_count critical-level entries — investigate"
    fail_count=$((fail_count + 1))
    ssh "$HOST" "docker logs hmats-engine --since 30m 2>&1 | grep -E 'CRITICAL|FATAL|Traceback' | tail -5" \
      2>/dev/null | sed 's/^/        /'
  fi

  # 4. Tick liveness --------------------------------------------------
  echo ""
  echo "${CYAN}[4] TICK LIVENESS${RESET}"
  local last_tick
  last_tick=$(ssh "$HOST" \
    "docker logs hmats-engine --since 15m 2>&1 | grep -E 'LIVE_DATA|AGENT-TRACE' | tail -1" 2>/dev/null)
  if [[ -z "$last_tick" ]]; then
    echo "  $WARN  no LIVE_DATA / AGENT-TRACE in last 15m"
  else
    local ts
    ts=$(printf '%s' "$last_tick" | sed -E 's/^([0-9-]+ [0-9:]+).*/\1/')
    echo "  $PASS  last tick: $ts"
    printf '  %s%s%s\n' "${DIM}" \
      "$(printf '%s' "$last_tick" | sed -E 's/^[0-9-]+ [0-9:]+ \| [^|]+ \| [A-Z]+\s*\| //; s/(.{120}).*/\1.../')" \
      "${RESET}"
  fi

  # 5. Fusion behavior ------------------------------------------------
  echo ""
  echo "${CYAN}[5] FUSION (DECIDE pool behavior, last 1h)${RESET}"
  local consensus conflict solo abstain total
  consensus=$(ssh_count 'DECIDE_CONSENSUS' '1h')
  conflict=$(ssh_count 'DECIDE_CONFLICT' '1h')
  solo=$(ssh_count 'DECIDE_SOLO' '1h')
  abstain=$(ssh_count 'DECIDE_ABSTAIN' '1h')
  total=$((consensus + conflict + solo + abstain))
  echo "  CONSENSUS=$consensus  CONFLICT=$conflict  SOLO=$solo  ABSTAIN=$abstain  (total=$total)"
  if [[ "$total" -eq 0 ]]; then
    echo "  $WARN  no fusion decisions in last 1h (engine warming up, or no DECIDE pool entered yet)"
  elif [[ "$conflict" -gt 0 && "$consensus" -eq 0 && "$solo" -eq 0 ]]; then
    echo "  $WARN  fusion is 100% conflict — agents persistently disagree"
  else
    echo "  $PASS  fusion engine producing decisions"
  fi

  # 6. Exit-DRL state -------------------------------------------------
  echo ""
  echo "${CYAN}[6] EXIT-DRL STATE${RESET}"
  local edrl_status
  edrl_status=$(ssh "$HOST" "docker logs hmats-engine 2>&1 | grep 'ExitDRLAgent:' | tail -1" 2>/dev/null)
  if [[ -z "$edrl_status" ]]; then
    echo "  $WARN  no ExitDRLAgent: line found"
  else
    if [[ "$edrl_status" == *"ready=NONE"* ]]; then
      echo "  $FAIL  ready=NONE — checkpoints failed to load"
      fail_count=$((fail_count + 1))
    elif [[ "$edrl_status" == *"EXIT_ONLY"* ]]; then
      echo "  $PASS  agent loaded, EXIT_ONLY mode active"
    else
      echo "  $WARN  agent loaded but mode is not EXIT_ONLY"
    fi
    printf '  %s%s%s\n' "${DIM}" \
      "$(printf '%s' "$edrl_status" | sed -E 's/^[0-9-]+ [0-9:]+ \| [^|]+ \| [A-Z]+\s*\| //; s/(.{140}).*/\1.../')" \
      "${RESET}"
  fi

  local demote_count
  demote_count=$(ssh_count 'DEMOTED to SHADOW' '1h')
  if [[ "$demote_count" -eq 0 ]]; then
    echo "  $PASS  no kill-switch demotions in last 1h"
  else
    echo "  $FAIL  $demote_count kill-switch demotions in last 1h — agent misbehaving"
    fail_count=$((fail_count + 1))
    ssh "$HOST" "docker logs hmats-engine --since 1h 2>&1 | grep 'DEMOTED to SHADOW' | tail -3" \
      2>/dev/null | sed 's/^/        /'
  fi

  local bridge_count
  bridge_count=$(ssh_count 'EXIT_DRL_BRIDGE' '24h')
  echo "  ${DIM}EXIT_DRL_BRIDGE fires (24h): $bridge_count${RESET}"

  # 7. Trades + alpha-gate activity -----------------------------------
  echo ""
  echo "${CYAN}[7] TRADE ACTIVITY (last 24h)${RESET}"
  local fills alpha_pass alpha_fail
  # [P62 2026-04-25] Engine logs `[FILL-QUALITY]` once per fill (and `[P0_EXECUTE]`
  # paired but at the same timestamp). Old pattern `[FILL]|FILLED|SCALE-OUT`
  # matched ZERO real fills (verified 2026-04-25: 3 actual fills at 08:26-28
  # but old grep returned 0). Use FILL-QUALITY only — one per fill.
  fills=$(ssh_count '\[FILL-QUALITY\]|\[SCALE-OUT\]' '24h')
  alpha_pass=$(ssh_count 'ALPHA_GATE: PASS' '24h')
  alpha_fail=$(ssh_count 'GATE_REJECT' '24h')
  echo "  fills=$fills  alpha_gate_pass=$alpha_pass  gate_reject=$alpha_fail"
  if [[ "$fills" -gt 0 ]]; then
    echo "  $PASS  trades executed in last 24h"
  elif [[ "$alpha_pass" -gt 0 ]]; then
    # Probe POST_DECIDE_VETO reasons to surface the actual blocker.
    local weekend_vetos ac2_vetos notional_vetos other_post_vetos
    weekend_vetos=$(ssh_count 'POST_DECIDE_VETO.*WEEKEND' '24h')
    ac2_vetos=$(ssh_count 'AC2_RATE_LIMITED|AC2_GLOBAL_RATE_LIMITED' '24h')
    notional_vetos=$(ssh_count 'NOTIONAL_TOO_SMALL|VC-5.*NOTIONAL' '24h')
    other_post_vetos=$(ssh_count 'POST_DECIDE_VETO' '24h')
    if [[ "$weekend_vetos" -gt 0 ]]; then
      echo "  $WARN  $weekend_vetos weekend-confidence vetoes — Saturday/Sunday with conf < 50% (intended)"
    elif [[ "$ac2_vetos" -gt 0 ]]; then
      echo "  $WARN  $ac2_vetos AC-2 rate-limit blocks — already at 2-fills/asset/24h ceiling"
    elif [[ "$notional_vetos" -gt 0 ]]; then
      echo "  $WARN  $notional_vetos notional-too-small blocks — VC-5 floor or sizing issue"
    elif [[ "$other_post_vetos" -gt 0 ]]; then
      echo "  $WARN  $other_post_vetos POST_DECIDE_VETO blocks (other) — investigate downstream"
    else
      echo "  $WARN  alpha gate passed but no fills — check tranche/size_cap downstream"
    fi
  elif [[ "$alpha_fail" -gt 0 ]]; then
    echo "  ${DIM}info: $alpha_fail gate rejects — system holding (correct in QUIET regimes)${RESET}"
  fi

  # 8. Resource usage -------------------------------------------------
  echo ""
  echo "${CYAN}[8] RESOURCE USAGE${RESET}"
  local stats_out
  stats_out=$(ssh "$HOST" \
    "docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}' | grep hmats" \
    2>/dev/null)
  while IFS='|' read -r name cpu mem mempct; do
    [[ -z "$name" ]] && continue
    local mem_int=${mempct%%.*}
    mem_int=${mem_int%\%}
    if [[ "$mem_int" =~ ^[0-9]+$ ]] && [[ "$mem_int" -gt 90 ]]; then
      echo "  $WARN  $name  cpu=$cpu  mem=$mem ($mempct)"
    else
      echo "  $PASS  $name  cpu=$cpu  mem=$mem ($mempct)"
    fi
  done <<< "$stats_out"

  # 9. Causal chain — module → flag → agent → authority -------------
  echo ""
  echo "${CYAN}[9] CAUSAL CHAIN HEALTH${RESET}"
  # Quick proxy: did engine startup show STUB / NOT_AVAILABLE / DISABLED warnings
  # at module-init time? If yes, agents are degraded silently.
  local stub_count not_avail_count disabled_count
  stub_count=$(ssh_count 'STUB \(' 'startup')
  not_avail_count=$(ssh_count 'unavailable \(|not.*available' '30m')
  disabled_count=$(ssh_count 'mode=DISABLED|DISABLED for this asset' '30m')
  # Note: --since startup is invalid for docker logs; use 24h as proxy for "this lifetime"
  if [[ "$stub_count" -eq 0 ]]; then stub_count=$(ssh_count 'STUB \(' '24h'); fi
  echo "  module STUB warnings (24h):       $stub_count"
  echo "  unavailable / not-available (30m): $not_avail_count"
  echo "  per-asset DISABLED (30m):          $disabled_count"
  # Fee-tier cliff warning
  local fee_cliff
  fee_cliff=$(ssh_count 'FEE_TIER_CLIFF' '30m')
  if [[ "$fee_cliff" -gt 0 ]]; then
    echo "  $WARN  $fee_cliff [FEE_TIER_CLIFF] warnings — monthly volume near \$10K boundary; alpha threshold will 6× when crossed"
  fi
  if [[ "$stub_count" -gt 0 ]]; then
    echo "  $WARN  $stub_count modules in STUB mode — likely degraded agent"
    fail_count=$((fail_count + 1))
  elif [[ "$not_avail_count" -gt 5 ]]; then
    echo "  $WARN  $not_avail_count 'unavailable' messages — possible silent degradation; run scripts/hmats_causal_audit"
  else
    echo "  $PASS  no module degradation signals"
  fi

  # Summary -----------------------------------------------------------
  echo ""
  echo "${CYAN}────────────────────────────────────────────────────────────────────${RESET}"
  if [[ "$fail_count" -eq 0 ]]; then
    echo "  ${GREEN}OVERALL: HEALTHY${RESET} (0 failures)"
  else
    echo "  ${RED}OVERALL: $fail_count FAILURE(S)${RESET}"
  fi
  echo "${CYAN}════════════════════════════════════════════════════════════════════${RESET}"

  return "$fail_count"
}

if [[ "$WATCH" -eq 1 ]]; then
  while true; do
    clear
    snapshot || true
    echo ""
    echo "${DIM}refresh in 60s — Ctrl-C to stop${RESET}"
    sleep 60
  done
else
  snapshot
  exit $?
fi
