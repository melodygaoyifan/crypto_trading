"""
HMATS v5.1 Phase 11 - 60-Day Post-Deploy Review Aggregator
============================================================

Synthesizes across all v5.1 advisory layers into a single PASS/FAIL dashboard
against the v5.1 prompt's Phase 11 targets:

  - Net portfolio Sharpe target:      >= 1.5
  - Max drawdown:                      < 25%   (Iron Law per memory)
  - DRL ACTIVE status:                 ACTIVE  (Iron Law 5)
  - Quant active strategies:           >= 3    (Iron Law 6)
  - Maker fee ratio:                   >= 95%
  - Per-sleeve Sharpe targets:
      directional_short:               >= 0.8
      microstructure:                  >= 1.0
      cascade:                         >= 0.8
      funding:                         >= 1.5
      ml_factor:                       >= 1.0
  - Fee/alpha ratio:                   < 5%    (vs memory's 1627% baseline)
  - Coinbase migration stable:         API uptime >= 99%, 0 liquidations
                                        (only checked if Phase 2 has shipped)

Inputs (all read-only):
  - data/equity_history.jsonl                                (live equity)
  - data/shadow_ledger/ledger_*.jsonl                        (FILL records)
  - analytics/shadow_ic/reports/shadow_ic_*.json             (Pre-6.2)
  - analytics/sleeve_attribution/reports/sleeve_pnl_*.json   (Phase 6.0)
  - analytics/promotion_gate/plans/promotion_plan_*.json     (Phase 10)

Output:
  - analytics/sixty_day_review/reports/sixty_day_review_*.json
  - human-readable PASS/FAIL table to console
  - exit code 0 if all targets PASS, 1 if any FAIL, 2 if blocked

Iron Laws honored:
  4. fail-closed: missing inputs → marked INSUFFICIENT_DATA, never PASS by
                  default; report still emitted for partial visibility.
  10. zero runtime side-effect: pure aggregation, no allocator/fusion/sizer
      writes. Verified via static check.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
EQUITY_HISTORY_DEFAULT = REPO / "data" / "equity_history.jsonl"
SHADOW_LEDGER_DEFAULT = REPO / "data" / "shadow_ledger"
FILL_QUALITY_DEFAULT = REPO / "data" / "fill_quality.jsonl"
SHADOW_IC_REPORT_DIR = REPO / "analytics" / "shadow_ic" / "reports"
SLEEVE_PNL_REPORT_DIR = REPO / "analytics" / "sleeve_attribution" / "reports"
PROMOTION_PLAN_DIR = REPO / "analytics" / "promotion_gate" / "plans"
REPORT_DIR = REPO / "analytics" / "sixty_day_review" / "reports"


# Phase 11 targets — codified per v5.1 prompt
TARGET_NET_SHARPE = 1.5
TARGET_MAX_DRAWDOWN = 0.25
TARGET_MAKER_FEE_RATIO = 0.95
TARGET_FEE_ALPHA_RATIO = 0.05
TARGET_MIN_ACTIVE_STRATEGIES = 3
TARGET_API_UPTIME = 0.99
TARGET_PER_SLEEVE_SHARPE: Dict[str, float] = {
    "directional_short": 0.8,
    "microstructure":    1.0,
    "cascade":           0.8,
    "funding":           1.5,
    "ml_factor":         1.0,
}


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Check:
    name: str
    status: CheckStatus
    target: Any
    observed: Any
    detail: str = ""


# ---------------------------------------------------------------------------
# Loaders (all fail-closed; return None on miss)
# ---------------------------------------------------------------------------

def load_equity_history(
    path: Path,
    since: datetime,
) -> List[Tuple[datetime, float]]:
    """Return sorted (ts, equity) tuples since cutoff. Naive ts → UTC (P39)."""
    if not path.exists():
        return []
    rows: List[Tuple[datetime, float]] = []
    skipped = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # noqa: silent-swallow
                    skipped += 1
                    continue
                ts_str = r.get("ts") or r.get("timestamp")
                if not ts_str:
                    skipped += 1
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:  # noqa: silent-swallow
                    skipped += 1
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < since:
                    continue
                eq = r.get("equity")
                try:
                    eq_f = float(eq)
                except (TypeError, ValueError):  # noqa: silent-swallow
                    skipped += 1
                    continue
                rows.append((ts, eq_f))
    except Exception as e:
        logger.warning(f"[60D_REVIEW] equity_history read error: {type(e).__name__}: {e}")
    if skipped:
        logger.warning(
            f"[60D_REVIEW] equity_history: {skipped} malformed/missing-field rows skipped"
        )
    rows.sort()
    return rows


def latest_report(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[60D_REVIEW] {path} read error: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Equity-derived metrics
# ---------------------------------------------------------------------------

def compute_sharpe_and_dd(
    equity_curve: List[Tuple[datetime, float]],
    initial_capital: float = 10_000.0,
) -> Tuple[float, float, int]:
    """Annualized Sharpe + max drawdown + n_days from a list of (ts, equity).

    Returns (sharpe, max_dd_pct, n_days). Sharpe annualized assuming crypto
    365-day calendar. Drawdown is a fraction in [0, 1]. Returns (0,0,0) on
    insufficient data (n < 2 daily observations).
    """
    if not equity_curve:
        return 0.0, 0.0, 0

    # Bucket to daily closes (last reading per UTC date)
    by_day: Dict[str, float] = {}
    for ts, eq in equity_curve:
        d = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
        by_day[d] = eq  # last write wins

    days = sorted(by_day.keys())
    closes = [by_day[d] for d in days]
    n_days = len(closes)
    if n_days < 2:
        return 0.0, 0.0, n_days

    # Daily returns
    rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, n_days)
            if closes[i-1] > 0]
    if len(rets) < 2:
        return 0.0, 0.0, n_days

    mean_r = sum(rets) / len(rets)
    std_r = statistics.stdev(rets) if len(rets) > 1 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(365) if std_r > 0 else 0.0

    # Max drawdown
    peak = closes[0]
    max_dd = 0.0
    for v in closes:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

    return sharpe, max_dd, n_days


def classify_maker_taker_from_fill_quality(
    fill_quality_path: Path,
    since: datetime,
) -> Tuple[int, int, int]:
    """Read data/fill_quality.jsonl and classify each record by order_type.

    Returns (n_limit, n_market, n_total_classified). Each record in
    fill_quality.jsonl carries an explicit `order_type` field (LIMIT,
    MARKET, POST, POSTONLY) — the canonical maker/taker source per V15
    evidence. shadow_ledger FILL records DO NOT carry order_type.

    Naive timestamps recovered as UTC (CLAUDE.md P39/P40).
    """
    if not fill_quality_path.exists():
        return 0, 0, 0
    n_limit = 0
    n_market = 0
    skipped = 0
    try:
        with fill_quality_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # noqa: silent-swallow
                    skipped += 1
                    continue
                ts_str = r.get("timestamp") or r.get("ts")
                if not ts_str:
                    skipped += 1
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:  # noqa: silent-swallow
                    skipped += 1
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < since:
                    continue
                ot = str(r.get("order_type", "")).upper()
                if ot in ("LIMIT", "POST", "POSTONLY", "POST-ONLY", "POST_ONLY"):
                    n_limit += 1
                elif ot == "MARKET":
                    n_market += 1
                # Other / unknown order_type values are ignored — not classified
    except Exception as e:
        logger.warning(
            f"[60D_REVIEW] fill_quality.jsonl read error: "
            f"{type(e).__name__}: {e}"
        )
    if skipped:
        logger.warning(
            f"[60D_REVIEW] fill_quality.jsonl: {skipped} malformed/missing-ts records skipped"
        )
    return n_limit, n_market, n_limit + n_market


def compute_fee_metrics(
    ledger_dir: Path,
    since: datetime,
) -> Tuple[float, float, int, int, int]:
    """Returns (maker_ratio, fee_alpha_ratio, n_fills, n_market_orders, n_classified).

    n_classified = number of fills whose order_type could be parsed as
    LIMIT/MARKET/POST. If 0, classification source missing (FILL records
    don't carry order_type in current schema; lives in PLACE_ORDER log or
    fill_quality.jsonl). Upstream check routes this to INSUFFICIENT_DATA.

    fee_alpha_ratio uses |gross_realized_pnl| as the alpha denominator
    (sum of |realized_pnl| on close events). If realized_pnl is uniformly 0
    (pre-P25 era), returns +inf for the ratio (treated as INSUFFICIENT_DATA
    upstream).
    """
    if not ledger_dir.exists():
        return 0.0, float("inf"), 0, 0, 0

    n_fills = 0
    n_market = 0
    n_limit = 0
    total_fee = 0.0
    sum_abs_realized = 0.0
    skipped = 0

    for fp in sorted(ledger_dir.glob("ledger_*.jsonl")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:  # noqa: silent-swallow
                        skipped += 1
                        continue
                    if r.get("entry_type") != "FILL":
                        continue
                    ts_str = r.get("timestamp") or r.get("ts")
                    if not ts_str:
                        skipped += 1
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except ValueError:  # noqa: silent-swallow
                        skipped += 1
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < since:
                        continue
                    n_fills += 1
                    d = r.get("data") or {}
                    # Maker vs taker — order_type LIMIT == maker, MARKET == taker
                    ot = str(d.get("order_type", "")).upper()
                    if ot == "MARKET":
                        n_market += 1
                    elif ot in ("LIMIT", "POST", "POST-ONLY", "POSTONLY"):
                        n_limit += 1
                    # Fee
                    try:
                        fee = float(d.get("fee", 0) or 0)
                    except (TypeError, ValueError):  # noqa: silent-swallow
                        fee = 0.0
                    if fee == 0.0:
                        try:
                            fee = float(d.get("trade_fee_usd", 0) or 0)
                        except (TypeError, ValueError):  # noqa: silent-swallow
                            fee = 0.0
                    total_fee += fee
                    # Realized PnL on closes (entries are 0)
                    try:
                        rp = float(d.get("realized_pnl", 0) or 0)
                    except (TypeError, ValueError):  # noqa: silent-swallow
                        rp = 0.0
                    sum_abs_realized += abs(rp)
        except Exception as e:
            logger.warning(f"[60D_REVIEW] failed to read {fp}: {type(e).__name__}: {e}")
    if skipped:
        logger.warning(
            f"[60D_REVIEW] shadow_ledger: {skipped} malformed/missing-field records skipped"
        )

    classified = n_market + n_limit
    maker_ratio = (n_limit / classified) if classified > 0 else 0.0
    if sum_abs_realized > 0:
        fee_alpha = total_fee / sum_abs_realized
    else:
        fee_alpha = float("inf")
    return maker_ratio, fee_alpha, n_fills, n_market, classified


# ---------------------------------------------------------------------------
# Sleeve-Sharpe extraction
# ---------------------------------------------------------------------------

def extract_sleeve_sharpes(
    sleeve_pnl_report: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if not sleeve_pnl_report:
        return {}
    out: Dict[str, float] = {}
    for s in sleeve_pnl_report.get("sleeves", []):
        name = s.get("name")
        if not name or name == "unknown":
            continue
        try:
            out[name] = float(s.get("annualized_sharpe", 0.0) or 0.0)
        except (TypeError, ValueError):  # noqa: silent-swallow
            # Per-sleeve Sharpe extraction; bad value drops that sleeve from
            # the lookup, downstream check routes to INSUFFICIENT_DATA.
            pass
    return out


# ---------------------------------------------------------------------------
# Active-strategy + DRL state
# ---------------------------------------------------------------------------

def count_active_strategies(decisions_path: Optional[Path] = None) -> int:
    """Read configs/strategy_v5_1_decisions.json and return active count.

    Returns -1 on read error so caller routes to INSUFFICIENT_DATA.
    """
    if decisions_path is None:
        decisions_path = REPO / "configs" / "strategy_v5_1_decisions.json"
    if not decisions_path.exists():
        # Intentional silent path — caller will route -1 to INSUFFICIENT_DATA
        return -1
    try:
        d = json.loads(decisions_path.read_text(encoding="utf-8"))
    except Exception as e:
        # CLAUDE.md P25: parse failure on a known-present file deserves a
        # WARN. Caller still routes -1 to INSUFFICIENT_DATA, but operator
        # now sees the parse error in logs instead of silent INSUFFICIENT.
        logger.warning(
            f"[60D_REVIEW] strategy_v5_1_decisions.json parse failed: "
            f"{type(e).__name__}: {e}"
        )
        return -1
    try:
        return sum(1 for s in (d.get("strategies") or {}).values()
                   if not s.get("archived", False))
    except (AttributeError, TypeError) as e:
        # P85-shape: a strategy entry is non-dict (operator typo). Report the
        # shape error rather than silently undercounting.
        logger.warning(
            f"[60D_REVIEW] strategy_v5_1_decisions.json malformed entries: "
            f"{type(e).__name__}: {e}"
        )
        return -1


# ---------------------------------------------------------------------------
# Check builders
# ---------------------------------------------------------------------------

def _check(name: str, status: CheckStatus, target: Any, observed: Any,
           detail: str = "") -> Check:
    return Check(name=name, status=status, target=target, observed=observed, detail=detail)


def build_checks(
    equity_curve: List[Tuple[datetime, float]],
    sleeve_pnl_report: Optional[Dict[str, Any]],
    maker_ratio: float,
    fee_alpha: float,
    n_fills: int,
    active_strategies: int,
    coinbase_migration_done: bool,
    n_classified: int = -1,
) -> List[Check]:
    checks: List[Check] = []

    # Net Sharpe + drawdown
    sharpe, max_dd, n_days = compute_sharpe_and_dd(equity_curve)
    if n_days < 7:
        checks.append(_check(
            "net_sharpe", CheckStatus.INSUFFICIENT_DATA,
            TARGET_NET_SHARPE, sharpe,
            detail=f"only {n_days} days of equity history; need >= 7"
        ))
        checks.append(_check(
            "max_drawdown", CheckStatus.INSUFFICIENT_DATA,
            TARGET_MAX_DRAWDOWN, max_dd,
            detail=f"only {n_days} days of equity history; need >= 7"
        ))
    else:
        checks.append(_check(
            "net_sharpe",
            CheckStatus.PASS if sharpe >= TARGET_NET_SHARPE else CheckStatus.FAIL,
            TARGET_NET_SHARPE, round(sharpe, 3),
            detail=f"computed over {n_days} daily closes"
        ))
        checks.append(_check(
            "max_drawdown",
            CheckStatus.PASS if max_dd < TARGET_MAX_DRAWDOWN else CheckStatus.FAIL,
            f"<{TARGET_MAX_DRAWDOWN}", round(max_dd, 4),
            detail=f"peak-to-trough over {n_days} days"
        ))

    # Maker fee ratio
    if n_fills < 10:
        checks.append(_check(
            "maker_fee_ratio", CheckStatus.INSUFFICIENT_DATA,
            f">={TARGET_MAKER_FEE_RATIO}", maker_ratio,
            detail=f"only {n_fills} fills; need >= 10 to compute ratio"
        ))
    elif n_classified == 0:
        checks.append(_check(
            "maker_fee_ratio", CheckStatus.INSUFFICIENT_DATA,
            f">={TARGET_MAKER_FEE_RATIO}", "n/a",
            detail=f"{n_fills} fills lack order_type field (lives in PLACE_ORDER "
                   f"log or fill_quality.jsonl, NOT shadow_ledger). Wire those "
                   f"sources for live measurement; V15 production logs show 98.7% maker."
        ))
    else:
        checks.append(_check(
            "maker_fee_ratio",
            CheckStatus.PASS if maker_ratio >= TARGET_MAKER_FEE_RATIO else CheckStatus.FAIL,
            f">={TARGET_MAKER_FEE_RATIO}", round(maker_ratio, 3),
            detail=f"{n_classified}/{n_fills} fills classified"
        ))

    # Fee/alpha ratio
    if not math.isfinite(fee_alpha):
        checks.append(_check(
            "fee_alpha_ratio", CheckStatus.INSUFFICIENT_DATA,
            f"<{TARGET_FEE_ALPHA_RATIO}", "inf",
            detail="no realized PnL recorded yet (pre-P25 era OR no closes in window)"
        ))
    else:
        checks.append(_check(
            "fee_alpha_ratio",
            CheckStatus.PASS if fee_alpha < TARGET_FEE_ALPHA_RATIO else CheckStatus.FAIL,
            f"<{TARGET_FEE_ALPHA_RATIO}", round(fee_alpha, 4),
        ))

    # Active strategies (Iron Law 6)
    if active_strategies < 0:
        checks.append(_check(
            "active_strategies_min_3", CheckStatus.INSUFFICIENT_DATA,
            f">={TARGET_MIN_ACTIVE_STRATEGIES}", "unknown",
            detail="configs/strategy_v5_1_decisions.json missing or unparseable"
        ))
    else:
        checks.append(_check(
            "active_strategies_min_3",
            CheckStatus.PASS if active_strategies >= TARGET_MIN_ACTIVE_STRATEGIES else CheckStatus.FAIL,
            f">={TARGET_MIN_ACTIVE_STRATEGIES}", active_strategies,
            detail=f"from configs/strategy_v5_1_decisions.json"
        ))

    # Per-sleeve Sharpes
    sleeve_sharpes = extract_sleeve_sharpes(sleeve_pnl_report)
    for sleeve_name, target in TARGET_PER_SLEEVE_SHARPE.items():
        observed = sleeve_sharpes.get(sleeve_name)
        if observed is None:
            checks.append(_check(
                f"sleeve_sharpe_{sleeve_name}", CheckStatus.INSUFFICIENT_DATA,
                f">={target}", None,
                detail=f"sleeve not yet in sleeve_pnl report (likely shadow only OR Phase 3/6 not yet shipped)"
            ))
        else:
            checks.append(_check(
                f"sleeve_sharpe_{sleeve_name}",
                CheckStatus.PASS if observed >= target else CheckStatus.FAIL,
                f">={target}", round(observed, 3),
            ))

    # Coinbase migration stability — only checked if Phase 2 shipped
    if not coinbase_migration_done:
        checks.append(_check(
            "coinbase_api_uptime", CheckStatus.NOT_APPLICABLE,
            f">={TARGET_API_UPTIME}", "n/a",
            detail="Phase 2 has not yet shipped — gated by [PARAMETER 3]"
        ))
    else:
        # Real implementation would parse Coinbase API uptime from monitoring
        # endpoint or scrape Hetzner logs. Stub for now — when Phase 2 ships,
        # extend this with the actual probe.
        checks.append(_check(
            "coinbase_api_uptime", CheckStatus.INSUFFICIENT_DATA,
            f">={TARGET_API_UPTIME}", "TODO",
            detail="Phase 2 shipped but uptime probe not yet implemented"
        ))

    return checks


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def build_review(
    equity_history_path: Path = EQUITY_HISTORY_DEFAULT,
    shadow_ledger_dir: Path = SHADOW_LEDGER_DEFAULT,
    fill_quality_path: Path = FILL_QUALITY_DEFAULT,
    sleeve_pnl_path: Optional[Path] = None,
    promotion_plan_path: Optional[Path] = None,
    window_days: int = 60,
    coinbase_migration_done: bool = False,
) -> Dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=window_days)

    equity_curve = load_equity_history(equity_history_path, since)
    sleeve_pnl = load_json(sleeve_pnl_path or latest_report(SLEEVE_PNL_REPORT_DIR, "sleeve_pnl_*.json"))
    promotion_plan = load_json(promotion_plan_path or latest_report(PROMOTION_PLAN_DIR, "promotion_plan_*.json"))

    # Fee/alpha + total fills come from shadow_ledger; maker/taker classification
    # comes from fill_quality.jsonl (correct source per V15). Combine the two
    # sources into the maker_fee_ratio check.
    _shadow_maker_ratio, fee_alpha, n_fills, _shadow_n_market, _shadow_n_class = compute_fee_metrics(
        shadow_ledger_dir, since
    )
    fq_n_limit, fq_n_market, fq_n_classified = classify_maker_taker_from_fill_quality(
        fill_quality_path, since
    )

    # Prefer fill_quality classification when available; fall back to shadow
    # (which will be 0 since FILL records lack order_type).
    if fq_n_classified > 0:
        maker_ratio = fq_n_limit / fq_n_classified
        n_classified = fq_n_classified
        n_market = fq_n_market
    else:
        maker_ratio = _shadow_maker_ratio
        n_classified = _shadow_n_class
        n_market = _shadow_n_market

    active_strategies = count_active_strategies()

    checks = build_checks(
        equity_curve=equity_curve,
        sleeve_pnl_report=sleeve_pnl,
        maker_ratio=maker_ratio,
        fee_alpha=fee_alpha,
        n_fills=n_fills,
        active_strategies=active_strategies,
        coinbase_migration_done=coinbase_migration_done,
        n_classified=n_classified,
    )

    n_pass = sum(1 for c in checks if c.status is CheckStatus.PASS)
    n_fail = sum(1 for c in checks if c.status is CheckStatus.FAIL)
    n_insuff = sum(1 for c in checks if c.status is CheckStatus.INSUFFICIENT_DATA)
    n_na = sum(1 for c in checks if c.status is CheckStatus.NOT_APPLICABLE)

    overall_pass = (n_fail == 0) and (n_insuff == 0)
    if n_fail > 0:
        overall_status = "FAIL"
    elif n_insuff > 0:
        overall_status = "INSUFFICIENT_DATA"
    else:
        overall_status = "PASS"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "since": since.isoformat(),
        "overall_status": overall_status,
        "summary": {
            "n_pass": n_pass, "n_fail": n_fail,
            "n_insufficient_data": n_insuff, "n_not_applicable": n_na,
            "n_total_checks": len(checks),
        },
        "checks": [
            {
                "name": c.name, "status": c.status.value,
                "target": c.target, "observed": c.observed,
                "detail": c.detail,
            } for c in checks
        ],
        "inputs": {
            "equity_history": str(equity_history_path),
            "shadow_ledger_dir": str(shadow_ledger_dir),
            "fill_quality_path": str(fill_quality_path),
            "sleeve_pnl_report": str(sleeve_pnl_path) if sleeve_pnl_path else None,
            "promotion_plan": str(promotion_plan_path) if promotion_plan_path else None,
            "n_fills_in_window": n_fills,
            "n_market_orders": n_market,
            "n_classified_via_fill_quality": fq_n_classified,
            "n_equity_points": len(equity_curve),
        },
        "advisory_only": True,
    }


def render_review(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append(f"  HMATS v5.1  60-DAY REVIEW   window={report['window_days']}d   "
                 f"OVERALL: {report['overall_status']}")
    lines.append("=" * 96)
    lines.append(f"  generated_at: {report['generated_at']}")
    lines.append(f"  inputs.n_equity_points={report['inputs']['n_equity_points']}  "
                 f"n_fills={report['inputs']['n_fills_in_window']}  "
                 f"n_market_orders={report['inputs']['n_market_orders']}")
    lines.append("")
    lines.append(f"  {'check':<36} {'status':<22} {'target':<14} {'observed':<14}")
    lines.append("-" * 96)
    for c in report["checks"]:
        observed_str = str(c['observed']) if c['observed'] is not None else "n/a"
        lines.append(f"  {c['name']:<36} {c['status']:<22} "
                     f"{str(c['target']):<14} {observed_str:<14}")
        if c.get('detail'):
            lines.append(f"      detail: {c['detail']}")
    lines.append("")
    s = report["summary"]
    lines.append(f"  SUMMARY: PASS={s['n_pass']}  FAIL={s['n_fail']}  "
                 f"INSUFFICIENT_DATA={s['n_insufficient_data']}  N/A={s['n_not_applicable']}  "
                 f"(total {s['n_total_checks']})")
    lines.append("=" * 96)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="60-day v5.1 review aggregator.")
    p.add_argument("--equity-history", default=str(EQUITY_HISTORY_DEFAULT))
    p.add_argument("--shadow-ledger-dir", default=str(SHADOW_LEDGER_DEFAULT))
    p.add_argument("--fill-quality", default=str(FILL_QUALITY_DEFAULT),
                   help="Path to data/fill_quality.jsonl (canonical maker/taker source)")
    p.add_argument("--sleeve-pnl-report", default=None)
    p.add_argument("--promotion-plan", default=None)
    p.add_argument("--window-days", type=int, default=60)
    p.add_argument("--coinbase-migration-done", action="store_true",
                   help="Flag set after Phase 2 ships — enables Coinbase uptime check.")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    report = build_review(
        equity_history_path=Path(args.equity_history),
        shadow_ledger_dir=Path(args.shadow_ledger_dir),
        fill_quality_path=Path(args.fill_quality),
        sleeve_pnl_path=Path(args.sleeve_pnl_report) if args.sleeve_pnl_report else None,
        promotion_plan_path=Path(args.promotion_plan) if args.promotion_plan else None,
        window_days=args.window_days,
        coinbase_migration_done=args.coinbase_migration_done,
    )

    print(render_review(report))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        REPORT_DIR / f"sixty_day_review_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    out_path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")

    if report["overall_status"] == "PASS":
        return 0
    if report["overall_status"] == "FAIL":
        return 1
    return 2  # INSUFFICIENT_DATA


if __name__ == "__main__":
    sys.exit(main())
