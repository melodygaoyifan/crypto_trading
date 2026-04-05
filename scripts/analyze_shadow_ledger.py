"""
Analyze shadow ledger: match FILL records to closest INTENT records,
compute estimated alpha, classify JUSTIFIED trades, and build alpha curve.
"""
import json
import glob
import os
from datetime import datetime, timedelta
from collections import defaultdict

# --- Config ---
LEDGER_DIR = r"c:\Users\melod\Downloads\hmats\data\shadow_ledger"
FRICTION_BPS = {"BTC": 19, "ETH": 21, "SOL": 26}
JUSTIFIED_RATIO = 2.0

def parse_timestamp(ts_str):
    """Parse ISO timestamp string."""
    return datetime.fromisoformat(ts_str)

def compute_alpha_est(direction, confidence):
    """
    alpha_est = abs(direction) * 200 * (0.7*confidence + 0.3*0.5) * 0.75
    """
    return abs(direction) * 200 * (0.7 * confidence + 0.3 * 0.5) * 0.75

def load_all_records():
    """Load all records from all ledger files."""
    files = sorted(glob.glob(os.path.join(LEDGER_DIR, "ledger_*.jsonl")))
    intents = []
    fills = []
    positions = []

    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = rec.get("entry_type", "")
                if entry_type == "INTENT":
                    intents.append(rec)
                elif entry_type == "FILL":
                    fills.append(rec)
                elif entry_type == "POSITION":
                    positions.append(rec)

    return intents, fills, positions

def match_fill_to_intent(fill, intents_by_asset):
    """Find closest INTENT for same asset within 60 seconds."""
    asset = fill["asset"]
    fill_ts = parse_timestamp(fill["timestamp"])

    candidates = intents_by_asset.get(asset, [])
    best = None
    best_delta = timedelta(seconds=999999)

    for intent in candidates:
        intent_ts = parse_timestamp(intent["timestamp"])
        delta = abs(fill_ts - intent_ts)

        # Within same session preferred, but allow 60s window
        if delta < best_delta and delta <= timedelta(seconds=60):
            # Prefer same session_id
            best = intent
            best_delta = delta

    # If no match within 60s, try same session_id with broader window
    if best is None:
        fill_session = fill.get("session_id", "")
        fill_tick = fill.get("tick_id", -1)
        for intent in candidates:
            if intent.get("session_id") == fill_session and intent.get("tick_id") == fill_tick:
                intent_ts = parse_timestamp(intent["timestamp"])
                delta = abs(fill_ts - intent_ts)
                if best is None or delta < best_delta:
                    best = intent
                    best_delta = delta

    return best

def main():
    intents, fills, positions = load_all_records()

    print(f"Total records: {len(intents)} INTENTs, {len(fills)} FILLs, {len(positions)} POSITIONs")
    print()

    # Index intents by asset for fast lookup
    intents_by_asset = defaultdict(list)
    for intent in intents:
        intents_by_asset[intent["asset"]].append(intent)

    # Sort intents by timestamp for each asset
    for asset in intents_by_asset:
        intents_by_asset[asset].sort(key=lambda x: x["timestamp"])

    # Match each fill to closest intent
    matched_pairs = []
    unmatched_fills = []

    for fill in fills:
        intent = match_fill_to_intent(fill, intents_by_asset)
        if intent:
            matched_pairs.append((fill, intent))
        else:
            unmatched_fills.append(fill)

    print(f"Matched: {len(matched_pairs)} fill-intent pairs")
    print(f"Unmatched fills: {len(unmatched_fills)}")
    print()

    # Compute alpha for all matched pairs
    all_trades = []
    for fill, intent in matched_pairs:
        asset = fill["asset"]
        fill_data = fill["data"]
        intent_data = intent["data"]

        direction = intent_data["direction"]
        confidence = intent_data["confidence"]
        side = fill_data.get("side", "?")
        price = fill_data.get("price", 0)
        size = fill_data.get("size", 0)
        fee = fill_data.get("fee", 0)
        realized_pnl = fill_data.get("realized_pnl", 0)
        notional = fill_data.get("notional", 0)

        # If notional is 0, compute from price*size
        if notional == 0 and price > 0:
            notional = price * size

        alpha_est_bps = compute_alpha_est(direction, confidence)
        friction_bps = FRICTION_BPS[asset]
        ratio = alpha_est_bps / friction_bps if friction_bps > 0 else 0

        # Realized alpha in bps
        realized_bps = (realized_pnl / notional * 10000) if notional > 0 else 0

        trade = {
            "timestamp": fill["timestamp"],
            "asset": asset,
            "side": side,
            "price": price,
            "size": size,
            "notional": notional,
            "fee": fee,
            "direction": direction,
            "confidence": confidence,
            "alpha_est_bps": alpha_est_bps,
            "friction_bps": friction_bps,
            "ratio": ratio,
            "realized_pnl": realized_pnl,
            "realized_bps": realized_bps,
            "justified": ratio >= JUSTIFIED_RATIO,
            "regime": intent_data.get("regime", "?"),
            "session_id": fill.get("session_id", "?"),
            "tick_id": fill.get("tick_id", "?"),
        }
        all_trades.append(trade)

    # ============================================
    # SECTION 1: ALL FILLS SUMMARY
    # ============================================
    print("=" * 120)
    print("SECTION 1: ALL FILLS SUMMARY")
    print("=" * 120)
    print(f"Total fills analyzed: {len(all_trades)}")

    # Count by asset
    by_asset = defaultdict(int)
    for t in all_trades:
        by_asset[t["asset"]] += 1
    for asset, count in sorted(by_asset.items()):
        print(f"  {asset}: {count} fills")

    # Count justified
    justified = [t for t in all_trades if t["justified"]]
    print(f"\nJUSTIFIED (ratio >= {JUSTIFIED_RATIO}): {len(justified)}")
    not_justified = [t for t in all_trades if not t["justified"]]
    print(f"NOT JUSTIFIED: {len(not_justified)}")
    print()

    # ============================================
    # SECTION 2: JUSTIFIED TRADES TABLE
    # ============================================
    print("=" * 120)
    print(f"SECTION 2: ALL {len(justified)} JUSTIFIED TRADES (alpha_est >= 2x friction)")
    print("=" * 120)

    # Header
    header = f"{'#':>3} {'Date':>19} {'Asset':>5} {'Side':>5} {'Price':>12} {'Notional':>12} {'Dir':>7} {'Conf':>6} " \
             f"{'Alpha_Est':>10} {'Friction':>9} {'Ratio':>6} {'Fee':>10} {'Real_PnL':>10} {'Real_bps':>10} {'Regime':>22}"
    print(header)
    print("-" * len(header))

    for i, t in enumerate(sorted(justified, key=lambda x: x["timestamp"]), 1):
        dt = t["timestamp"][:19]
        print(f"{i:>3} {dt:>19} {t['asset']:>5} {t['side']:>5} {t['price']:>12.2f} {t['notional']:>12.2f} "
              f"{t['direction']:>7.3f} {t['confidence']:>6.3f} {t['alpha_est_bps']:>10.2f} {t['friction_bps']:>9d} "
              f"{t['ratio']:>6.2f} {t['fee']:>10.2f} {t['realized_pnl']:>10.4f} {t['realized_bps']:>10.4f} {t['regime']:>22}")

    # Summary stats for justified
    print()
    if justified:
        avg_alpha = sum(t["alpha_est_bps"] for t in justified) / len(justified)
        avg_real_pnl = sum(t["realized_pnl"] for t in justified) / len(justified)
        avg_real_bps = sum(t["realized_bps"] for t in justified) / len(justified)
        total_pnl = sum(t["realized_pnl"] for t in justified)
        total_fees = sum(t["fee"] for t in justified)
        total_notional = sum(t["notional"] for t in justified)
        wins = sum(1 for t in justified if t["realized_pnl"] > 0)
        losses = sum(1 for t in justified if t["realized_pnl"] < 0)
        zeros = sum(1 for t in justified if t["realized_pnl"] == 0)

        print(f"  JUSTIFIED SUMMARY:")
        print(f"    Avg alpha_est_bps: {avg_alpha:.2f}")
        print(f"    Avg realized_pnl:  ${avg_real_pnl:.4f}")
        print(f"    Avg realized_bps:  {avg_real_bps:.4f}")
        print(f"    Total realized_pnl: ${total_pnl:.4f}")
        print(f"    Total fees:         ${total_fees:.2f}")
        print(f"    Total notional:     ${total_notional:.2f}")
        print(f"    Wins/Losses/Zero:   {wins}/{losses}/{zeros}")

    # ============================================
    # SECTION 3: ALPHA CURVE (all trades by bucket)
    # ============================================
    print()
    print("=" * 120)
    print("SECTION 3: ALPHA CURVE - Estimated vs Realized Alpha by Bucket")
    print("=" * 120)

    buckets = [
        (0, 10, "[0-10]"),
        (10, 20, "[10-20]"),
        (20, 30, "[20-30]"),
        (30, 40, "[30-40]"),
        (40, 50, "[40-50]"),
        (50, 100, "[50-100]"),
        (100, 999, "[100+]"),
    ]

    bucket_header = f"{'Bucket':>10} {'Count':>6} {'Avg Alpha_Est':>14} {'Avg Real_PnL':>13} {'Avg Real_bps':>13} {'Win Rate':>9} {'Total PnL':>11} {'Total Notional':>15}"
    print(bucket_header)
    print("-" * len(bucket_header))

    for lo, hi, label in buckets:
        bucket_trades = [t for t in all_trades if lo <= t["alpha_est_bps"] < hi]
        if not bucket_trades:
            print(f"{label:>10} {0:>6} {'-':>14} {'-':>13} {'-':>13} {'-':>9} {'-':>11} {'-':>15}")
            continue

        count = len(bucket_trades)
        avg_alpha = sum(t["alpha_est_bps"] for t in bucket_trades) / count
        avg_rpnl = sum(t["realized_pnl"] for t in bucket_trades) / count
        avg_rbps = sum(t["realized_bps"] for t in bucket_trades) / count
        total_pnl = sum(t["realized_pnl"] for t in bucket_trades)
        total_notional = sum(t["notional"] for t in bucket_trades)

        # Win rate: only count trades with non-zero realized_pnl
        non_zero = [t for t in bucket_trades if t["realized_pnl"] != 0]
        if non_zero:
            wins = sum(1 for t in non_zero if t["realized_pnl"] > 0)
            win_rate = wins / len(non_zero) * 100
            wr_str = f"{win_rate:.0f}%"
        else:
            wr_str = "N/A"

        print(f"{label:>10} {count:>6} {avg_alpha:>14.2f} {avg_rpnl:>13.4f} {avg_rbps:>13.4f} {wr_str:>9} {total_pnl:>11.4f} {total_notional:>15.2f}")

    # ============================================
    # SECTION 4: CALIBRATION ANALYSIS
    # ============================================
    print()
    print("=" * 120)
    print("SECTION 4: MAX_ALPHA_BPS CALIBRATION")
    print("=" * 120)

    # Only use trades with nonzero realized_pnl for calibration
    calibration_trades = [t for t in all_trades if t["realized_pnl"] != 0 and t["notional"] > 0]

    if calibration_trades:
        avg_alpha_est = sum(t["alpha_est_bps"] for t in calibration_trades) / len(calibration_trades)
        avg_realized_bps = sum(t["realized_bps"] for t in calibration_trades) / len(calibration_trades)

        print(f"  Trades with nonzero realized_pnl: {len(calibration_trades)}")
        print(f"  Avg estimated alpha (bps):  {avg_alpha_est:.4f}")
        print(f"  Avg realized alpha (bps):   {avg_realized_bps:.4f}")

        if abs(avg_realized_bps) > 0.0001:
            overestimate_ratio = avg_alpha_est / avg_realized_bps
            print(f"  Overestimate ratio:         {overestimate_ratio:.2f}x")
            calibrated_max = 200 / abs(overestimate_ratio)
            print(f"  Current MAX_ALPHA_BPS:      200")
            print(f"  Calibrated MAX_ALPHA_BPS:   {calibrated_max:.1f}")
        else:
            print(f"  Avg realized alpha too close to zero for ratio calculation")
            print(f"  This suggests alpha estimate is severely disconnected from reality")
    else:
        print("  No trades with nonzero realized_pnl found for calibration")

    # ============================================
    # SECTION 5: PER-ASSET BREAKDOWN
    # ============================================
    print()
    print("=" * 120)
    print("SECTION 5: PER-ASSET ALPHA ANALYSIS")
    print("=" * 120)

    for asset in ["BTC", "ETH", "SOL"]:
        asset_trades = [t for t in all_trades if t["asset"] == asset]
        cal_trades = [t for t in asset_trades if t["realized_pnl"] != 0 and t["notional"] > 0]

        if not asset_trades:
            continue

        print(f"\n  {asset}:")
        print(f"    Total fills: {len(asset_trades)}")
        print(f"    Fills with realized_pnl != 0: {len(cal_trades)}")

        avg_est = sum(t["alpha_est_bps"] for t in asset_trades) / len(asset_trades)
        print(f"    Avg alpha_est (all): {avg_est:.2f} bps")

        if cal_trades:
            avg_est_cal = sum(t["alpha_est_bps"] for t in cal_trades) / len(cal_trades)
            avg_real = sum(t["realized_bps"] for t in cal_trades) / len(cal_trades)
            total_pnl = sum(t["realized_pnl"] for t in cal_trades)
            wins = sum(1 for t in cal_trades if t["realized_pnl"] > 0)

            print(f"    Avg alpha_est (w/ pnl): {avg_est_cal:.2f} bps")
            print(f"    Avg realized_bps: {avg_real:.4f} bps")
            print(f"    Total realized_pnl: ${total_pnl:.4f}")
            print(f"    Win rate: {wins}/{len(cal_trades)} = {wins/len(cal_trades)*100:.0f}%")

            if abs(avg_real) > 0.0001:
                ratio = avg_est_cal / avg_real
                print(f"    Overestimate ratio: {ratio:.2f}x")

    # ============================================
    # SECTION 6: DETAILED NON-ZERO PNL TRADES
    # ============================================
    print()
    print("=" * 120)
    print("SECTION 6: ALL TRADES WITH NON-ZERO REALIZED PNL")
    print("=" * 120)

    nonzero = [t for t in all_trades if t["realized_pnl"] != 0]
    nonzero.sort(key=lambda x: x["timestamp"])

    header2 = f"{'#':>3} {'Date':>19} {'Asset':>5} {'Side':>5} {'Price':>10} {'Notional':>12} {'Alpha_Est':>10} " \
              f"{'Real_PnL':>12} {'Real_bps':>10} {'Ratio':>6} {'Justified':>10}"
    print(header2)
    print("-" * len(header2))

    for i, t in enumerate(nonzero, 1):
        dt = t["timestamp"][:19]
        j_str = "YES" if t["justified"] else "no"
        print(f"{i:>3} {dt:>19} {t['asset']:>5} {t['side']:>5} {t['price']:>10.2f} {t['notional']:>12.2f} "
              f"{t['alpha_est_bps']:>10.2f} {t['realized_pnl']:>12.4f} {t['realized_bps']:>10.4f} {t['ratio']:>6.2f} {j_str:>10}")

    # ============================================
    # SECTION 7: DISTRIBUTION OF ALPHA ESTIMATES
    # ============================================
    print()
    print("=" * 120)
    print("SECTION 7: DISTRIBUTION OF ALPHA ESTIMATES ACROSS ALL FILLS")
    print("=" * 120)

    alphas = sorted([t["alpha_est_bps"] for t in all_trades])
    print(f"  Total fills: {len(alphas)}")
    print(f"  Min alpha_est: {min(alphas):.2f} bps")
    print(f"  Max alpha_est: {max(alphas):.2f} bps")
    print(f"  Mean alpha_est: {sum(alphas)/len(alphas):.2f} bps")
    print(f"  Median alpha_est: {alphas[len(alphas)//2]:.2f} bps")

    # Percentiles
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        idx = int(len(alphas) * pct / 100)
        idx = min(idx, len(alphas) - 1)
        print(f"  P{pct}: {alphas[idx]:.2f} bps")

    # Count how many above various thresholds
    for thresh in [19, 21, 26, 38, 42, 52]:
        above = sum(1 for a in alphas if a >= thresh)
        print(f"  >= {thresh} bps: {above} ({above/len(alphas)*100:.1f}%)")

if __name__ == "__main__":
    main()