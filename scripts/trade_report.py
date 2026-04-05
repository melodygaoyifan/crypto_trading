import json, os, glob
from datetime import datetime
from collections import defaultdict

LEDGER_DIR = os.path.normpath("c:/Users/melod/Downloads/hmats/data/shadow_ledger")
TRANCHE_FILE = os.path.normpath("c:/Users/melod/Downloads/hmats/data/tranche_state.json")
POSITIONS_FILE = os.path.normpath("c:/Users/melod/Downloads/hmats/data/paper_positions.json")

entries = []
for f in sorted(glob.glob(os.path.join(LEDGER_DIR, "ledger_*.jsonl"))):
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except:
                    pass

fills = [e for e in entries if e.get("entry_type") == "FILL"]
intents = [e for e in entries if e.get("entry_type") == "INTENT"]
positions = [e for e in entries if e.get("entry_type") == "POSITION"]

with open(TRANCHE_FILE, "r") as f:
    tranche = json.load(f)

with open(POSITIONS_FILE, "r") as f:
    paper_pos = json.load(f)

SEP = "=" * 120
DASH = "-" * 120

print(SEP)
print("HMATS PAPER TRADING REPORT")
print(SEP)
print("Data range: {} to {}".format(entries[0]["timestamp"][:10], entries[-1]["timestamp"][:10]))
print("Total entries: {} (Fills: {}, Intents: {}, Position changes: {})".format(
    len(entries), len(fills), len(intents), len(positions)))
print()

print(SEP)
print("TABLE 1: ALL FILL RECORDS")
print(SEP)
print("{:>3} {:>10} {:>8} {:>4} {:>4} {:>12} {:>12} {:>8} {:>10} {:>12}".format(
    "#", "Date", "Time(UTC)", "Asset", "Side", "Size", "Price", "Fee($)", "PnL($)", "Notional"))
print(DASH)
for i, fl in enumerate(fills, 1):
    d = fl["data"]
    ts = fl["timestamp"]
    pnl = d.get("realized_pnl", 0)
    pnl_str = "{:+.4f}".format(pnl) if pnl != 0 else "0.0000"
    print("{:>3} {:>10} {:>8} {:>4} {:>4} {:>12.6f} {:>12.2f} {:>8.2f} {:>10} {:>12.2f}".format(
        i, ts[:10], ts[11:19], fl["asset"], d["side"], d["size"], d["price"], d["fee"],
        pnl_str, d.get("notional", 0)))
print()

print(SEP)
print("TABLE 2: ALL INTENT RECORDS (Strategy Signals)")
print(SEP)
print("{:>3} {:>10} {:>8} {:>4} {:>7} {:>10} {:>6} {:>6} {:>25}".format(
    "#", "Date", "Time(UTC)", "Asset", "Dir", "Size", "Entry?", "Conf", "Regime"))
print(DASH)
for i, e in enumerate(intents, 1):
    d = e["data"]
    ts = e["timestamp"]
    entry = "YES" if d.get("is_entry") else "no"
    regime = d.get("regime", "N/A")
    if len(regime) > 25:
        regime = regime[:22] + "..."
    print("{:>3} {:>10} {:>8} {:>4} {:>+7.3f} {:>10.4f} {:>6} {:>6.3f} {:>25}".format(
        i, ts[:10], ts[11:19], e["asset"], d["direction"], d["size"], entry,
        d.get("confidence", 0), regime))
print()

print(SEP)
print("TABLE 3: POSITION CHANGES")
print(SEP)
print("{:>3} {:>10} {:>8} {:>4} {:>10} {:>10} {:>7} {:>7} {:>10} {:>60}".format(
    "#", "Date", "Time(UTC)", "Asset", "Old_Size", "New_Size", "Old_Dir", "New_Dir",
    "PnL($)", "Reason (truncated)"))
print(DASH)
for i, e in enumerate(positions, 1):
    d = e["data"]
    ts = e["timestamp"]
    reason = d.get("reason", "")[:60]
    pnl = d.get("realized_pnl", 0)
    pnl_str = "{:+.4f}".format(pnl) if pnl != 0 else "0.0000"
    print("{:>3} {:>10} {:>8} {:>4} {:>10.6f} {:>10.6f} {:>7} {:>7} {:>10} {:>60}".format(
        i, ts[:10], ts[11:19], e["asset"], d["old_size"], d["new_size"],
        d["old_direction"], d["new_direction"], pnl_str, reason))
print()

print(SEP)
print("TABLE 4: PER-ASSET SUMMARY")
print(SEP)

for asset in ["BTC", "ETH", "SOL"]:
    af = [fl for fl in fills if fl["asset"] == asset]
    ai = [e for e in intents if e["asset"] == asset]
    total_fee = sum(fl["data"]["fee"] for fl in af)
    total_pnl = sum(fl["data"].get("realized_pnl", 0) for fl in af)
    total_notional = sum(fl["data"].get("notional", 0) for fl in af)
    buys = sum(1 for fl in af if fl["data"]["side"] == "BUY")
    sells = sum(1 for fl in af if fl["data"]["side"] == "SELL")

    regime_counts = defaultdict(int)
    for e in ai:
        regime_counts[e["data"].get("regime", "N/A")] += 1

    tr = tranche.get(asset, {})
    pp = paper_pos.get("positions", {}).get(asset, {})

    print()
    print("--- {} ---".format(asset))
    print("  Fills: {} (BUY={}, SELL={})".format(len(af), buys, sells))
    print("  Total Notional:     ${:>12,.2f}".format(total_notional))
    print("  Total Fees:         ${:>12,.4f}".format(total_fee))
    print("  Total Realized PnL: ${:>12,.4f}".format(total_pnl))
    print("  Intents: {}".format(len(ai)))
    print("  Regimes: {}".format(dict(regime_counts)))
    print("  Tranche State: Level={}, Dir={}, Entry=${:,.2f}, Unrealized={:+.1f}bps".format(
        tr.get("level", "N/A"), tr.get("direction", "N/A"),
        tr.get("entry_price", 0), tr.get("unrealized_pnl_bps", 0)))
    print("  Paper Position: Exp={:.4f}, Dir={}, Strategy={}, Entry=${:,.2f}".format(
        pp.get("exposure", 0), pp.get("direction", 0),
        pp.get("strategy", "N/A"), pp.get("entry_price", 0)))
    if pp.get("entry_time"):
        print("  Entry Time: {}".format(pp["entry_time"]))
print()

print(SEP)
print("TABLE 5: DAILY PnL BREAKDOWN")
print(SEP)
daily = defaultdict(lambda: defaultdict(lambda: {"fills": 0, "fee": 0.0, "pnl": 0.0, "notional": 0.0}))
for fl in fills:
    dk = fl["timestamp"][:10]
    asset = fl["asset"]
    daily[dk][asset]["fills"] += 1
    daily[dk][asset]["fee"] += fl["data"]["fee"]
    daily[dk][asset]["pnl"] += fl["data"].get("realized_pnl", 0)
    daily[dk][asset]["notional"] += fl["data"].get("notional", 0)

print("{:>12} {:>5} {:>6} {:>12} {:>10} {:>10}".format(
    "Date", "Asset", "#Fills", "Notional", "Fee($)", "PnL($)"))
print("-" * 70)

cum_pnl = 0
cum_fee = 0
for dk in sorted(daily.keys()):
    day_pnl = 0
    day_fee = 0
    for asset in ["BTC", "ETH", "SOL"]:
        if asset in daily[dk]:
            d = daily[dk][asset]
            day_pnl += d["pnl"]
            day_fee += d["fee"]
            print("{:>12} {:>5} {:>6} {:>12,.2f} {:>10.2f} {:>+10.4f}".format(
                dk, asset, d["fills"], d["notional"], d["fee"], d["pnl"]))
    cum_pnl += day_pnl
    cum_fee += day_fee
    print("{:>12} {:>5} {:>6} {:>12} {:>10.2f} {:>+10.4f}  (cum: PnL={:+.4f}, Fee={:.2f})".format(
        "", "TOTAL", "", "", day_fee, day_pnl, cum_pnl, cum_fee))
    print()

print()
print("{:>18}: Realized PnL = ${:+.4f}, Total Fees = ${:,.2f}".format(
    "GRAND TOTAL", cum_pnl, cum_fee))
print("{:>18}  Net (PnL - Fees) = ${:+.2f}".format("", cum_pnl - cum_fee))

print()
print(SEP)
print("TABLE 6: CURRENT UNREALIZED P&L")
print(SEP)
print("{:>5} {:>9} {:>5} {:>12} {:>11} {:>12} {:>20}".format(
    "Asset", "Direction", "Level", "Entry$", "Current Exp", "Unreal(bps)", "Since"))
print("-" * 80)
for asset in ["BTC", "ETH", "SOL"]:
    tr = tranche.get(asset, {})
    dv = tr.get("direction", "N/A")
    lv = "T" + str(tr.get("level", "?"))
    ep = tr.get("entry_price", 0)
    ce = tr.get("current_exposure", 0)
    ubps = tr.get("unrealized_pnl_bps", 0)
    ea = tr.get("entered_at", "N/A")
    if ea != "N/A":
        ea = ea[:19]
    print("{:>5} {:>9} {:>5} {:>12,.2f} {:>11.4f} {:>+12.1f} {:>20}".format(
        asset, dv, lv, ep, ce, ubps, ea))

print()
print(SEP)
print("END OF REPORT")
print(SEP)
