"""
GMM Regime Classifier Live Diagnostic
Validates GMM models against recent Kraken data.
Checks: confidence distribution, flip rate, regime balance, distribution shift.
"""
import json, joblib, numpy as np, requests, time, sys
from pathlib import Path
from collections import Counter

base = Path(__file__).parent.parent / "models" / "regime_classifier"
PAIRS = {"BTC": "XXBTZUSD", "ETH": "XETHZUSD", "SOL": "SOLUSD"}
SPREAD_DEFAULTS = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}


def compute_features(closes, volumes, asset, np_mod):
    """Compute 12 GMM features from OHLCV arrays (mirrors _predict_gmm_regime)."""
    n = len(closes)
    rets = np_mod.diff(closes) / np_mod.where(closes[:-1] != 0, closes[:-1], 1.0)

    ret_4h = float(rets[-1]) if len(rets) > 0 else 0.0
    ret_1h = ret_4h / 4.0
    ret_24h = float((closes[-1] - closes[-7]) / closes[-7]) if n >= 7 and closes[-7] > 0 else 0.0
    ret_7d = float((closes[-1] - closes[-43]) / closes[-43]) if n >= 43 and closes[-43] > 0 else 0.0
    vol_1h = float(np_mod.std(rets[-2:])) if len(rets) >= 2 else 0.02
    vol_24h = float(np_mod.std(rets[-6:])) if len(rets) >= 6 else 0.02
    vol_pct = float(np_mod.searchsorted(np_mod.sort(volumes), volumes[-1]) / n * 100) if n > 0 else 50.0

    if len(rets) >= 30:
        _wv = np_mod.lib.stride_tricks.sliding_window_view(rets, 6)
        rolling_v = np_mod.std(_wv, axis=1)
        vov = float(np_mod.std(rolling_v[-20:])) if len(rolling_v) >= 20 else 0.0
    else:
        vov = 0.0

    if len(rets) >= 6:
        recent = rets[-6:]
        pos = int(np_mod.sum(recent > 0))
        neg = int(np_mod.sum(recent < 0))
        mom_con = (max(pos, neg) / 6.0) * (1 if pos >= neg else -1)
    else:
        mom_con = 0.0

    # RSI-14 approximation
    if len(rets) >= 14:
        gains = np_mod.where(rets[-14:] > 0, rets[-14:], 0)
        losses = np_mod.where(rets[-14:] < 0, -rets[-14:], 0)
        avg_gain = np_mod.mean(gains)
        avg_loss = np_mod.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi = 50.0

    return np_mod.array([
        ret_1h, ret_4h, ret_24h, ret_7d,
        vol_1h, vol_24h, vol_pct, vov,
        mom_con, 0.87, 100.0 - rsi, SPREAD_DEFAULTS[asset],
    ], dtype=np_mod.float64)


def predict(model, features, scaler_mean, scaler_scale):
    """Scale, clip, predict."""
    scaled = (features - scaler_mean) / scaler_scale
    n_extreme = int(np.sum(np.abs(scaled) > 3.0))
    scaled = np.clip(scaled, -4.0, 4.0)
    probs = model.predict_proba(scaled.reshape(1, -1))[0]
    idx = int(np.argmax(probs))
    return probs, idx, float(probs[idx]), n_extreme, scaled


def main():
    print("=" * 80)
    print("GMM REGIME CLASSIFIER LIVE DIAGNOSTIC")
    print("=" * 80)

    verdicts = []

    for asset in ["BTC", "ETH", "SOL"]:
        # Load model + config
        with open(base / asset / "gmm_config.json") as f:
            cfg = json.load(f)
        model = joblib.load(base / asset / "gmm_model.pkl")
        scaler_mean = np.array(cfg["scaler_mean"])
        scaler_scale = np.array(cfg["scaler_scale"])
        regime_names = cfg["regime_names"]
        feature_cols = cfg["feature_cols"]
        train_conf = cfg["mean_confidence"]
        train_flip = cfg["flip_rate"]
        train_weights = cfg["weights"]

        # Fetch 720 bars (~120 days) of 4H OHLCV from Kraken
        pair = PAIRS[asset]
        since = int(time.time()) - 720 * 14400
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240&since={since}"
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data.get("error"):
            print(f"\n{asset}: API error: {data['error']}")
            continue

        ohlc_key = [k for k in data["result"] if k != "last"][0]
        bars = data["result"][ohlc_key]
        closes = np.array([float(b[4]) for b in bars])
        volumes = np.array([float(b[6]) for b in bars])
        n = len(closes)

        # Current bar prediction
        features = compute_features(closes, volumes, asset, np)
        probs, idx, conf, n_extreme, scaled = predict(model, features, scaler_mean, scaler_scale)
        regime = regime_names[idx] if idx < len(regime_names) else f"R{idx}"

        print(f"\n{'='*40} {asset} {'='*40}")
        print(f"  Price: ${closes[-1]:,.2f} | Bars fetched: {n}")
        print(f"  Current regime: {regime} (conf={conf:.4f})")
        prob_str = " | ".join(f"{rn[:8]}={p:.3f}" for rn, p in zip(regime_names, probs))
        print(f"  Probabilities: {prob_str}")

        # Distribution shift
        shift_pct = n_extreme / 12 * 100
        shift_status = "OK" if n_extreme <= 1 else ("WARN" if n_extreme <= 3 else "CRITICAL")
        print(f"  Distribution shift: {n_extreme}/12 features |z|>3 ({shift_pct:.0f}%) [{shift_status}]")
        if n_extreme > 0:
            for i, (fname, zval) in enumerate(zip(feature_cols, scaled)):
                if abs(zval) > 3.0:
                    print(f"    {fname}: z={zval:+.2f} (raw={features[i]:.6f}, train_mean={scaler_mean[i]:.6f})")

        # Rolling diagnostic over last N bars
        window = min(50, n - 50)
        conf_history = []
        regime_history = []
        for bar_end in range(n - window, n):
            feat = compute_features(closes[:bar_end + 1], volumes[:bar_end + 1], asset, np)
            pr, ri, c, _, _ = predict(model, feat, scaler_mean, scaler_scale)
            conf_history.append(c)
            rn = regime_names[ri] if ri < len(regime_names) else f"R{ri}"
            regime_history.append(rn)

        # Flip rate
        flips = sum(1 for i in range(1, len(regime_history)) if regime_history[i] != regime_history[i - 1])
        live_flip = flips / max(1, len(regime_history) - 1)

        # Confidence stats
        live_conf_mean = np.mean(conf_history)
        live_conf_min = np.min(conf_history)

        print(f"\n  --- Rolling {len(regime_history)}-bar diagnostic ---")
        print(f"  Confidence:  live_mean={live_conf_mean:.3f}  train_mean={train_conf:.3f}  delta={live_conf_mean - train_conf:+.3f}")
        print(f"               live_min={live_conf_min:.3f}")
        print(f"  Flip rate:   live={live_flip:.1%}  train={train_flip:.1%}  delta={live_flip - train_flip:+.1%}")

        # Regime distribution
        regime_counts = Counter(regime_history)
        print(f"\n  Regime distribution (last {len(regime_history)} bars):")
        print(f"  {'Regime':<25s} {'Live%':>6s} {'Train%':>7s} {'Delta':>7s}  Flag")
        print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*7}  ----")
        for rn in regime_names:
            cnt = regime_counts.get(rn, 0)
            pct = cnt / len(regime_history) * 100
            tw = train_weights[regime_names.index(rn)] * 100
            delta = pct - tw
            flag = "<<<" if abs(delta) > 20 else ("!" if abs(delta) > 15 else "")
            print(f"  {rn:<25s} {pct:5.1f}% {tw:6.1f}% {delta:+6.1f}%  {flag}")

        # Verdict for this asset
        issues = []
        if live_conf_mean < train_conf - 0.08:
            issues.append(f"confidence drop ({live_conf_mean:.3f} vs {train_conf:.3f})")
        if live_conf_mean > train_conf + 0.08:
            issues.append(f"overconfidence ({live_conf_mean:.3f} vs {train_conf:.3f})")
        if live_flip - train_flip > 0.15:
            issues.append(f"regime instability (flip {live_flip:.0%} vs {train_flip:.0%})")
        if live_flip - train_flip < -0.20:
            issues.append(f"regime stickiness (flip {live_flip:.0%} vs {train_flip:.0%})")
        if n_extreme >= 3:
            issues.append(f"distribution shift ({n_extreme}/12 extreme features)")
        for rn in regime_names:
            cnt = regime_counts.get(rn, 0)
            pct = cnt / len(regime_history) * 100
            tw = train_weights[regime_names.index(rn)] * 100
            if abs(pct - tw) > 25:
                issues.append(f"{rn} imbalance ({pct:.0f}% vs train {tw:.0f}%)")

        status = "PASS" if not issues else "WARN"
        verdicts.append((asset, status, issues))

    # Final verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    all_pass = True
    for asset, status, issues in verdicts:
        if status == "PASS":
            print(f"  {asset}: PASS - GMM operating within training distribution")
        else:
            all_pass = False
            print(f"  {asset}: WARN")
            for iss in issues:
                print(f"    - {iss}")

    if all_pass:
        print("\n  OVERALL: GMM models are healthy. No retrain needed at this time.")
    else:
        print("\n  OVERALL: Some anomalies detected. Monitor closely; retrain if issues persist.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
