"""
Record manual Step 16 gating checks required before auto-entering Live Phase 1.

Usage:
    python -X utf8 scripts/mark_step16_manual_checks.py --risk-controls-pass --kill-switch-pass --note "verified on desk"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "manual_gates"
OUT_PATH = OUT_DIR / "step16_manual_checks.json"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write manual Step 16 gate checks.")
    parser.add_argument("--risk-controls-pass", action="store_true", help="Mark risk controls verification as passed.")
    parser.add_argument("--risk-controls-fail", action="store_true", help="Mark risk controls verification as failed.")
    parser.add_argument("--kill-switch-pass", action="store_true", help="Mark kill switch manual test as passed.")
    parser.add_argument("--kill-switch-fail", action="store_true", help="Mark kill switch manual test as failed.")
    parser.add_argument("--note", default="", help="Optional free-form note.")
    args = parser.parse_args(argv)

    if args.risk_controls_pass and args.risk_controls_fail:
        print("ERROR: choose only one of --risk-controls-pass / --risk-controls-fail")
        return 1
    if args.kill_switch_pass and args.kill_switch_fail:
        print("ERROR: choose only one of --kill-switch-pass / --kill-switch-fail")
        return 1

    out = {
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "risk_controls_verified": True if args.risk_controls_pass else (False if args.risk_controls_fail else None),
        "kill_switch_tested": True if args.kill_switch_pass else (False if args.kill_switch_fail else None),
        "note": args.note or "",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
