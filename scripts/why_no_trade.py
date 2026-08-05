#!/usr/bin/env python3
"""[P155] Answer "why is nothing trading?" from state ALREADY on disk — no waiting.

The `[HEALTH_T3] BLOCKED N consecutive ticks` alert says only "Check VETO_CHAIN
logs", which is a dead end whenever the real blocker is a collapsed
`target_exposure` (the veto chain is then silent). The patched T3 message names
the clause, but that requires a deploy plus a 4H tick.

This script needs neither: `_process_4h_tick_inner` writes
`data/diagnostics/diag_<asset>_<tick>.json` unconditionally on EVERY tick, and
its `engine_decide` probe records the exact three fields that
`TradeIntentV36.is_actionable` is built from. So the last tick's answer is
already sitting on the volume.

    is_actionable == (not veto_active) and |direction| > dir_thresh
                     and target_exposure > exp_thresh

Usage (on the server):
    docker exec hmats-engine python -X utf8 scripts/why_no_trade.py
    docker exec hmats-engine python -X utf8 scripts/why_no_trade.py --history 5

Locally against a copied volume:
    HMATS_DATA_DIR=/path/to/data python -X utf8 scripts/why_no_trade.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Mirrors integration_v36.TradeIntentV36.is_actionable defaults. The diag probe
# records only the three clause inputs, not the per-intent threshold overrides,
# so OPPORTUNITY-mode relaxations are reported as a caveat rather than guessed.
BASE_DIR_THRESHOLD = 0.10
BASE_EXP_THRESHOLD = 0.01
OPPORTUNITY_DIR_THRESHOLD = 0.05


def data_dir() -> str:
    return os.environ.get("HMATS_DATA_DIR", "data")


# ---------------------------------------------------------------------------
# diag parsing
# ---------------------------------------------------------------------------

def _extract(blob: str, key: str) -> Optional[str]:
    """Pull one key out of a repr()'d dict.

    `_diag_record` stores `repr(output)[:200]`, so the value is NOT JSON and may
    contain enum reprs like `<SystemMode.NORMAL: 'NORMAL'>` that would break
    ast.literal_eval. Regex per key is the robust read, and tolerates the
    200-char truncation dropping trailing keys.
    """
    m = re.search(rf"'{re.escape(key)}':\s*([^,}}]+)", blob)
    return m.group(1).strip() if m else None


def _as_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _as_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    if raw.startswith("True"):
        return True
    if raw.startswith("False"):
        return False
    return None


def load_latest_diags(root: str, per_asset: int) -> Dict[str, List[Tuple[str, dict]]]:
    """Newest-first diag files grouped by asset."""
    out: Dict[str, List[Tuple[str, dict]]] = {}
    pattern = os.path.join(root, "diagnostics", "diag_*.json")
    for path in sorted(glob.glob(pattern), reverse=True):
        name = os.path.basename(path)
        m = re.match(r"diag_([A-Za-z0-9]+)_", name)
        if not m:
            continue
        asset = m.group(1)
        bucket = out.setdefault(asset, [])
        if len(bucket) >= per_asset:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                bucket.append((name, json.load(fh)))
        except Exception as err:
            print(f"  ! unreadable {name}: {err}", file=sys.stderr)
    return out


def diagnose(diag: dict) -> Dict[str, Any]:
    """Decompose is_actionable for one tick's diag record."""
    comps = diag.get("components") or {}
    probe = comps.get("engine_decide") or {}
    blob = str(probe.get("output") or "")
    if not blob or blob == "None":
        return {"ok": False, "note": "engine_decide probe absent or empty"}

    veto = _as_bool(_extract(blob, "veto_active"))
    actionable = _as_bool(_extract(blob, "is_actionable"))
    direction = _as_float(_extract(blob, "direction"))
    exposure = _as_float(_extract(blob, "target_exposure"))
    mode_raw = _extract(blob, "system_mode") or ""
    mode = "OPPORTUNITY" if "OPPORTUNITY" in mode_raw else (
        "NORMAL" if "NORMAL" in mode_raw else mode_raw.strip("'\"<> "))

    dir_thresh = OPPORTUNITY_DIR_THRESHOLD if mode == "OPPORTUNITY" else BASE_DIR_THRESHOLD

    blockers: List[str] = []
    if veto:
        blockers.append("VETO_ACTIVE")
    if direction is not None and abs(direction) <= dir_thresh:
        blockers.append(f"WEAK_DIRECTION (|dir|={abs(direction):.4f} <= {dir_thresh:.2f})")
    if exposure is not None and exposure <= BASE_EXP_THRESHOLD:
        blockers.append(f"ZERO_EXPOSURE (target_exposure={exposure:.4f} <= {BASE_EXP_THRESHOLD:.2f})")

    return {
        "ok": True,
        "tick_time": diag.get("tick_time"),
        "actionable": actionable,
        "veto_active": veto,
        "direction": direction,
        "target_exposure": exposure,
        "mode": mode,
        "blockers": blockers,
    }


# ---------------------------------------------------------------------------
# secondary state
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def sleeve_and_routing(root: str) -> List[str]:
    """The other half: even a perfect intent trades nothing if the sleeve is halted."""
    lines: List[str] = []

    routing = _read_json(os.path.join(root, "coinbase_routing_state.json"))
    if routing is None:
        lines.append("routing state : ABSENT -> defaults to PRE_PHASE_2 (no assets routed)")
    else:
        lines.append(f"routing state : phase={routing.get('phase')} "
                     f"assets={routing.get('coinbase_assets')}")

    sleeve = _read_json(os.path.join(root, "coinbase_sleeve_state.json"))
    if sleeve is None:
        lines.append("sleeve state  : ABSENT -> baseline not yet anchored")
    else:
        halted = sleeve.get("halted")
        mark = "  <<< STICKY HALT — manual reset_halt() required" if halted else ""
        lines.append(f"sleeve state  : halted={halted} reason={sleeve.get('halt_reason') or '-'} "
                     f"baseline=${sleeve.get('sleeve_start_equity') or 0:,.2f}{mark}")

    pnl_path = os.path.join(root, "coinbase_sleeve_pnl.jsonl")
    if os.path.exists(pnl_path):
        try:
            with open(pnl_path, "r", encoding="utf-8") as fh:
                rows = [json.loads(x) for x in fh if x.strip()]
            if rows:
                last = rows[-1]
                lines.append(f"sleeve pnl    : {len(rows)} points, last equity="
                             f"${last.get('equity_usd', 0):,.2f} "
                             f"pnl=${last.get('pnl_usd', 0):+,.2f}")
        except Exception as err:
            lines.append(f"sleeve pnl    : unreadable ({err})")
    else:
        lines.append("sleeve pnl    : ABSENT -> log_pnl_point() has never run")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Why is nothing trading? (reads on-disk tick state)")
    ap.add_argument("--history", type=int, default=1,
                    help="ticks per asset to show, newest first (default 1)")
    ap.add_argument("--data-dir", default=None, help="override HMATS_DATA_DIR")
    args = ap.parse_args()

    root = args.data_dir or data_dir()
    print(f"=== why_no_trade — data dir: {os.path.abspath(root)} ===\n")

    diags = load_latest_diags(root, max(1, args.history))
    if not diags:
        print("No data/diagnostics/diag_*.json found.")
        print("Either the engine has not completed a tick, or HMATS_DATA_DIR is wrong.")
        print("On the server the engine's WORKDIR is /opt/hmats, so run this from there.")
        return 2

    verdicts: Dict[str, List[str]] = {}
    for asset in sorted(diags):
        print(f"--- {asset} ---")
        for fname, diag in diags[asset]:
            d = diagnose(diag)
            if not d.get("ok"):
                print(f"  {fname}: {d.get('note')}")
                continue
            print(f"  tick {d['tick_time']}  mode={d['mode']}")
            print(f"    actionable={d['actionable']}  veto_active={d['veto_active']}  "
                  f"dir={d['direction']}  target_exposure={d['target_exposure']}")
            if d["actionable"]:
                print("    -> ACTIONABLE (this tick was not blocked here)")
            elif d["blockers"]:
                print(f"    -> BLOCKED BY: {'; '.join(d['blockers'])}")
                verdicts.setdefault(asset, []).extend(d["blockers"])
            else:
                print("    -> BLOCKED, but no clause reproduces it "
                      "(threshold override or truncated probe)")
        print()

    print("--- Coinbase sleeve (the only venue authorised to trade since Phase B) ---")
    for line in sleeve_and_routing(root):
        print(f"  {line}")
    print()

    print("=== VERDICT ===")
    if not verdicts:
        print("  No blocked intents in the sampled ticks — the intent layer is NOT the")
        print("  bottleneck. Look at the sleeve state above and at [COINBASE-MANAGE].")
    else:
        for asset, bl in sorted(verdicts.items()):
            primary = sorted({b.split(" (")[0] for b in bl})
            print(f"  {asset}: {', '.join(primary)}")
        print()
        print("  ZERO_EXPOSURE   -> sizing collapsed. The veto chain is INNOCENT and silent;")
        print("                     look at the alpha gate / friction, not VETO_CHAIN logs.")
        print("  WEAK_DIRECTION  -> signal below the actionable floor; fusion, not execution.")
        print("  VETO_ACTIVE     -> now the veto chain genuinely is the place to look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
