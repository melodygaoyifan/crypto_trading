#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, glob, os
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# [P53 2026-04-25] repo-relative paths.
_REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = str(_REPO_ROOT / "data" / "shadow_ledger")
TRANCHE_FILE = str(_REPO_ROOT / "data" / "tranche_state.json")
JSONL_PATTERN = os.path.join(LEDGER_DIR, "ledger_*.jsonl")

def fmt_price(v):
    if v is None: return "N/A"
    if abs(v) >= 1000: return f"{v:,.2f}"
    elif abs(v) >= 1: return f"{v:,.4f}"
    else: return f"{v:.6f}"

def fmt_dollar(v):
    if v is None: return "N/A"
    D = "$"
    return f"{D}{v:,.4f}" if abs(v) < 10 else f"{D}{v:,.2f}"

def fmt_size(v):
    if v is None: return "N/A"
    return f"{v:.8f}" if abs(v) < 0.01 else f"{v:.6f}"

def truncate(s, maxlen=80):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen-3] + "..."

def parse_ts(ts_str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(ts_str, fmt)
        except ValueError: continue
    return datetime.min

def dlabel(d):
    if d > 0: return "LONG"
    elif d < 0: return "SHORT"
    return "FLAT"

NL = chr(10)
SEP = "=" * 140

def print_table(headers, rows, title=None):
    if title:
        print(NL + SEP)
        print("  " + title)
        print(SEP)
    if not rows:
        print("  (no records)")
        return
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        sr = [str(v) for v in row]
        str_rows.append(sr)
        for i, v in enumerate(sr):
            col_widths[i] = max(col_widths[i], len(v))
    align = []
    for i, h in enumerate(headers):
        nc = 0
        for sr in str_rows[:10]:
            v = sr[i].replace("$","").replace(",","").replace("%","").strip()
            try:
                float(v)
                nc += 1
            except ValueError: pass
        align.append(">" if nc > len(str_rows[:10]) / 2 else "<")
    hl = "  ".join(f"{h:{a}{w}}" for h,w,a in zip(headers, col_widths, align))
    print("  " + hl)
    sl = "  ".join("-"*w for w in col_widths)
    print("  " + sl)
    for sr in str_rows:
        rl = "  ".join(f"{v:{a}{w}}" for v,w,a in zip(sr, col_widths, align))
        print("  " + rl)
    print(f"  ({len(str_rows)} rows)")
# Parse all JSONL
fills = []
intents = []
positions = []

for filepath in sorted(glob.glob(JSONL_PATTERN)):
    with open(filepath, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            et = rec.get("entry_type","")
            ts_str = rec.get("timestamp","")
            asset = rec.get("asset","???")
            data = rec.get("data",{})
            tick_id = rec.get("tick_id","")
            sid = rec.get("session_id","")
            ts = parse_ts(ts_str)

            if et == "FILL":
                fills.append(dict(ts=ts, ts_str=ts_str, asset=asset,
                    side=data.get("side","?"), size=data.get("size",0),
                    price=data.get("price",0), fee=data.get("fee",0),
                    realized_pnl=data.get("realized_pnl",0),
                    order_id=data.get("order_id",""),
                    notional=data.get("notional",0),
                    tick_id=tick_id, session_id=sid))
            elif et == "INTENT":
                intents.append(dict(ts=ts, ts_str=ts_str, asset=asset,
                    direction=data.get("direction",0), size=data.get("size",0),
                    is_entry=data.get("is_entry",False),
                    confidence=data.get("confidence",0),
                    regime=data.get("regime","?"), reason=data.get("reason",""),
                    tick_id=tick_id, session_id=sid))
            elif et == "POSITION":
                positions.append(dict(ts=ts, ts_str=ts_str, asset=asset,
                    old_size=data.get("old_size",0), new_size=data.get("new_size",0),
                    old_direction=data.get("old_direction",0),
                    new_direction=data.get("new_direction",0),
                    realized_pnl=data.get("realized_pnl",0),
                    reason=data.get("reason",""),
                    tick_id=tick_id, session_id=sid))

fills.sort(key=lambda x: x["ts"])
intents.sort(key=lambda x: x["ts"])
positions.sort(key=lambda x: x["ts"])
# ===== TABLE 1: FILLS =====
fh = ["#","Date","Time(UTC)","Asset","Side","Size","Price","Notional","Fee","Realized_PnL","Order_ID"]
fr = []
for i, f in enumerate(fills, 1):
    fr.append([i, f["ts"].strftime("%Y-%m-%d"), f["ts"].strftime("%H:%M:%S"),
        f["asset"], f["side"], fmt_size(f["size"]), fmt_price(f["price"]),
        fmt_dollar(f["notional"]), fmt_dollar(f["fee"]),
        fmt_dollar(f["realized_pnl"]), f["order_id"]])
print_table(fh, fr, "TABLE 1: ALL FILL RECORDS (" + str(len(fills)) + " fills)")

# ===== TABLE 2: INTENTS =====
ih = ["#","Date","Time(UTC)","Asset","Direction","Size","Entry?","Confidence","Regime","Reason"]
ir_ = []
for i, it in enumerate(intents, 1):
    dv = it["direction"]
    dl = ("SHORT" if dv < 0 else "LONG") + " (" + f"{dv:+.4f}" + ")"
    conf = it["confidence"]
    ir_.append([i, it["ts"].strftime("%Y-%m-%d"), it["ts"].strftime("%H:%M:%S"),
        it["asset"], dl, fmt_size(it["size"]),
        "YES" if it["is_entry"] else "NO",
        f"{conf:.4f}", it["regime"],
        truncate(it["reason"],50) if it["reason"] else ""])
print_table(ih, ir_, "TABLE 2: ALL INTENT RECORDS (" + str(len(intents)) + " intents)")

# ===== TABLE 3: POSITION CHANGES =====
ph = ["#","Date","Time(UTC)","Asset","Old_Size","New_Size","Old_Dir","New_Dir","Realized_PnL","Reason"]
prr = []
for i, p_ in enumerate(positions, 1):
    prr.append([i, p_["ts"].strftime("%Y-%m-%d"), p_["ts"].strftime("%H:%M:%S"),
        p_["asset"], fmt_size(p_["old_size"]), fmt_size(p_["new_size"]),
        dlabel(p_["old_direction"]), dlabel(p_["new_direction"]),
        fmt_dollar(p_["realized_pnl"]),
        truncate(p_["reason"],80)])
print_table(ph, prr, "TABLE 3: POSITION CHANGES (" + str(len(positions)) + " changes)")
# ===== TABLE 4: SUMMARY PER ASSET =====
tranche_state = {}
try:
    with open(TRANCHE_FILE, "r", encoding="utf-8") as fh:
        tranche_state = json.load(fh)
except Exception as e:
    print("  WARN: tranche_state.json: " + str(e))

astats = defaultdict(lambda: dict(fills=0, buys=0, sells=0,
    total_fee=0.0, total_rpnl=0.0, total_notional=0.0,
    intents=0, entries=0, exits=0))
for f in fills:
    aa = f["asset"]
    astats[aa]["fills"] += 1
    astats[aa]["total_fee"] += f["fee"]
    astats[aa]["total_rpnl"] += f["realized_pnl"]
    astats[aa]["total_notional"] += f["notional"]
    if f["side"] == "BUY": astats[aa]["buys"] += 1
    else: astats[aa]["sells"] += 1
for it in intents:
    aa = it["asset"]
    astats[aa]["intents"] += 1
    if it["is_entry"]: astats[aa]["entries"] += 1
    else: astats[aa]["exits"] += 1

s4h = ["Asset","#Fills","#Buys","#Sells","#Intents","#Entries","#Exits",
    "Tot_Notional","Tot_Fee","Tot_Real_PnL",
    "Pos_Dir","Tranche","Entry_Price","Unreal(bps)"]
s4r = []
gf=0; gfe=0.0; grp=0.0; gn=0.0
for asset in sorted(astats.keys()):
    s = astats[asset]; t = tranche_state.get(asset,{})
    gf += s["fills"]; gfe += s["total_fee"]
    grp += s["total_rpnl"]; gn += s["total_notional"]
    dstr = t.get("direction","N/A")
    if isinstance(dstr, str): dstr = dstr.upper()
    ubps = t.get("unrealized_pnl_bps", 0)
    ep = t.get("entry_price", None)
    lv = t.get("level", "?")
    has_ubps = "unrealized_pnl_bps" in t
    s4r.append([asset, s["fills"], s["buys"], s["sells"],
        s["intents"], s["entries"], s["exits"],
        fmt_dollar(s["total_notional"]), fmt_dollar(s["total_fee"]),
        fmt_dollar(s["total_rpnl"]),
        dstr,
        "T" + str(lv),
        fmt_price(ep),
        f"{ubps:.1f}" if has_ubps else "N/A"])
s4r.append(["TOTAL", gf, "", "", "", "", "",
    fmt_dollar(gn), fmt_dollar(gfe), fmt_dollar(grp), "", "", "", ""])
print_table(s4h, s4r, "TABLE 4: SUMMARY PER ASSET")
# ===== TABLE 5: DAILY PnL =====
daily = defaultdict(lambda: defaultdict(lambda: dict(fills=0, fee=0.0, rpnl=0.0, notional=0.0)))
for f in fills:
    ds = f["ts"].strftime("%Y-%m-%d"); aa = f["asset"]
    daily[ds][aa]["fills"] += 1
    daily[ds][aa]["fee"] += f["fee"]
    daily[ds][aa]["rpnl"] += f["realized_pnl"]
    daily[ds][aa]["notional"] += f["notional"]

assets_sorted = sorted(astats.keys())
d5h = ["Date"]
for aa in assets_sorted:
    d5h.extend([aa+"_#", aa+"_Fee", aa+"_PnL"])
d5h.extend(["Tot_#", "Tot_Fee", "Tot_PnL", "Cum_PnL"])

d5r = []
cum_pnl = 0.0
for ds in sorted(daily.keys()):
    row = [ds]; df=0; dfe=0.0; dp=0.0
    for aa in assets_sorted:
        d = daily[ds][aa]
        row.append(d["fills"] if d["fills"]>0 else "")
        row.append(fmt_dollar(d["fee"]) if d["fills"]>0 else "")
        row.append(fmt_dollar(d["rpnl"]) if d["fills"]>0 else "")
        df += d["fills"]; dfe += d["fee"]; dp += d["rpnl"]
    cum_pnl += dp
    row.extend([df, fmt_dollar(dfe), fmt_dollar(dp), fmt_dollar(cum_pnl)])
    d5r.append(row)
trow = ["TOTAL"]
for aa in assets_sorted:
    s = astats[aa]
    trow.extend([s["fills"], fmt_dollar(s["total_fee"]), fmt_dollar(s["total_rpnl"])])
trow.extend([gf, fmt_dollar(gfe), fmt_dollar(grp), fmt_dollar(cum_pnl)])
d5r.append(trow)
print_table(d5h, d5r, "TABLE 5: DAILY PnL SUMMARY")
# ===== ADDITIONAL INSIGHTS =====
print(NL + SEP)
print("  ADDITIONAL INSIGHTS")
print(SEP)

rc = defaultdict(int)
for it in intents: rc[it["regime"]] += 1
print(NL + "  Regime distribution (from INTENT records):")
for regime, count in sorted(rc.items(), key=lambda x: -x[1]):
    bar = "#" * (count // 2)
    print("    " + f"{regime:<30s} {count:4d}  " + bar)

sessions = set()
for f in fills: sessions.add(f["session_id"])
for it in intents: sessions.add(it["session_id"])
print(NL + "  Total unique sessions: " + str(len(sessions)))

tids = set()
for f in fills:
    if f["tick_id"] is not None: tids.add(f["tick_id"])
if tids:
    print("  Tick ID range (fills): " + str(min(tids)) + " - " + str(max(tids)))

if fills:
    print("  First fill: " + fills[0]["ts_str"])
    print("  Last fill:  " + fills[-1]["ts_str"])
    delta = fills[-1]["ts"] - fills[0]["ts"]
    ddays = delta.days
    dhours = delta.seconds // 3600
    dmins = (delta.seconds % 3600) // 60
    print("  Time span:  " + str(ddays) + "d " + str(dhours) + "h " + str(dmins) + "m")

print(NL + "  Average fill metrics per asset:")
for aa in assets_sorted:
    af = [f for f in fills if f["asset"]==aa]
    if af:
        avg_not = sum(f["notional"] for f in af) / len(af)
        tn = sum(f["notional"] for f in af)
        avg_fee_bps = (sum(f["fee"] for f in af) / tn) * 10000 if tn > 0 else 0
        print("    " + aa + ": " + str(len(af)) + " fills, avg_notional=" + fmt_dollar(avg_not) + ", fee_rate=" + f"{avg_fee_bps:.1f}" + "bps")

print(NL + "  Current positions (from tranche_state.json):")
for asset in sorted(tranche_state.keys()):
    t = tranche_state[asset]
    d = t.get("direction","?")
    lv = t.get("level","?")
    ep = t.get("entry_price",0)
    ubps = t.get("unrealized_pnl_bps",0)
    ea = t.get("entered_at","?")
    exp_ = t.get("current_exposure",0)
    print("    " + asset + ": " + d.upper() + " T" + str(lv) + ", entry=" + fmt_price(ep) + ", exposure=" + f"{exp_:.6f}" + ", unrealized=" + f"{ubps:.1f}" + "bps, since=" + ea)

print(NL + SEP)
print("  END OF REPORT")
print(SEP)