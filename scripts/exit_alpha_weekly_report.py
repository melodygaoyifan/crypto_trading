from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from analytics.exit_alpha_tracker import ExitAlphaTracker


def build_report(
    *,
    tracker: Optional[ExitAlphaTracker] = None,
    start_at: Optional[str] = None,
    end_at: Optional[str] = None,
    lookback_days: int = 7,
    min_trades: int = 5,
) -> Dict[str, Any]:
    report_tracker = tracker or ExitAlphaTracker()
    report = report_tracker.weekly_review_dict(
        start_at=start_at,
        end_at=end_at,
        lookback_days=lookback_days,
        min_trades=min_trades,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build weekly exit-alpha report and bounded parameter recommendations.")
    parser.add_argument("--start", dest="start_at", default=None, help="UTC ISO8601 start timestamp")
    parser.add_argument("--end", dest="end_at", default=None, help="UTC ISO8601 end timestamp")
    parser.add_argument("--lookback-days", type=int, default=7, help="Lookback window when --start is omitted")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades required before emitting live recommendations")
    parser.add_argument(
        "--out",
        default="logs/exit_alpha/weekly_report.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    report = build_report(
        start_at=args.start_at,
        end_at=args.end_at,
        lookback_days=args.lookback_days,
        min_trades=args.min_trades,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
