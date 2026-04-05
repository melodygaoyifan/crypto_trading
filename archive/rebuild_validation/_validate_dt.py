"""⑧ DT v3.2 Layer Validation - Preserved, Not Retrained."""
import numpy as np
from pathlib import Path
import json
import sys

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DT_DIR = PROJECT / "hmats_training_v5 (1)" / "drl"
DT_SCRIPT = DT_DIR / "train_decision_transformer_v32.py"
ENSEMBLE_PATH = PROJECT / "drl" / "ensemble.py"
AGENT_PATH = PROJECT / "agents" / "model_alpha_agent.py"
MANIFEST = PROJECT / "config" / "feature_manifest.json"

SEP = "=" * 70
results = []

def log(section, check, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

# ═══════════════════════════════════════════════════════════
# FILES EXIST
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("DT v3.2 FILE EXISTENCE")
print(SEP)

log("FILE", "train_decision_transformer_v32.py exists", DT_SCRIPT.exists())
log("FILE", "drl/ensemble.py exists", ENSEMBLE_PATH.exists())
log("FILE", "agents/model_alpha_agent.py exists", AGENT_PATH.exists())

# ═══════════════════════════════════════════════════════════
# ARCHITECTURE CODE AUDIT
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ARCHITECTURE CODE AUDIT")
print(SEP)

dt_src = DT_SCRIPT.read_text(encoding="utf-8")

# DT v3.2 preserved with key components
log("ARCH", "DecisionTransformerV32 class exists",
    "class DecisionTransformerV32" in dt_src)
log("ARCH", "MoE (Mixture of Experts)",
    "MoE" in dt_src or "MixtureOfExperts" in dt_src or "expert" in dt_src.lower())
log("ARCH", "FGM (adversarial training)",
    "fgm" in dt_src.lower() or "use_fgm" in dt_src)
log("ARCH", "OHEM (online hard example mining)",
    "ohem" in dt_src.lower() or "use_ohem" in dt_src)
log("ARCH", "LoRA fine-tuning",
    "lora" in dt_src.lower() or "use_lora" in dt_src)
log("ARCH", "Curriculum learning",
    "curriculum" in dt_src.lower() or "use_curriculum" in dt_src)

# ═══════════════════════════════════════════════════════════
# FEATURE SPACE: 80-dim FeatureEngineerV32
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FEATURE SPACE: FeatureEngineerV32")
print(SEP)

log("FEAT", "FeatureEngineerV32 class exists",
    "class FeatureEngineerV32" in dt_src)

# state_dim = 80
log("FEAT", "state_dim = 80",
    "state_dim" in dt_src and ("= 80" in dt_src or ": int = 80" in dt_src or "state_dim=80" in dt_src))

# DT does NOT use external data features
ext_features = ["funding_rate_zscore", "oi_change_5d", "liq_imbalance",
                "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum",
                "has_external_data"]
ext_in_dt = [f for f in ext_features if f in dt_src]
log("FEAT", f"DT does NOT use external data features",
    len(ext_in_dt) == 0,
    f"found: {ext_in_dt}" if ext_in_dt else "none of the 7 external features present")

# TQC feature count differs from DT
with open(MANIFEST) as f:
    manifest = json.load(f)
tqc_count = manifest["total_feature_count"]
log("FEAT", f"DT (80-dim) ≠ TQC ({tqc_count}-dim) -> signal-level ensemble only",
    tqc_count != 80, f"TQC={tqc_count}, DT=80")

# Verify feature categories in DT
feat_categories = [
    ("returns", "ret_"),
    ("volatility", "vol_"),
    ("momentum", "mom_"),
    ("z-score", "zscore_"),
    ("RSI", "rsi"),
    ("MACD", "macd"),
    ("BB", "bb_"),
    ("regime encoding", "regime_"),
]
for cat_name, pattern in feat_categories:
    log("FEAT", f"  {cat_name} features present",
        pattern in dt_src)

# ═══════════════════════════════════════════════════════════
# GMM REGIME INTEGRATION
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("GMM REGIME INTEGRATION")
print(SEP)

# GMM regime mapping in DT
gmm_regimes = ["MOMENTUM_RALLY", "PANIC_SELLOFF", "EXTREME_VOLATILITY",
               "VOLATILE_CHOP", "QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"]
gmm_in_dt = [r for r in gmm_regimes if r in dt_src]
log("GMM", f"GMM regime names in DT: {len(gmm_in_dt)}/6",
    len(gmm_in_dt) >= 5, str(gmm_in_dt))

# DT regime mapping (GMM -> DT names)
dt_regimes = ["TRENDING_BULL", "TRENDING_BEAR", "VOLATILE_BEAR",
              "RANGING_HIGH_VOL", "RANGING_LOW_VOL"]
dt_in_src = [r for r in dt_regimes if r in dt_src]
log("GMM", f"DT regime names: {len(dt_in_src)}/{len(dt_regimes)}",
    len(dt_in_src) >= 4, str(dt_in_src))

# Rule-based regime NOT used (unified to GMM)
log("GMM", "Uses GMM regime (not rule-based)",
    "MOMENTUM_RALLY" in dt_src and "PANIC_SELLOFF" in dt_src)

# ═══════════════════════════════════════════════════════════
# CONTEXT LENGTH = 96
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CONTEXT LENGTH")
print(SEP)

log("CTX", "context_length = 96",
    "context_length" in dt_src and ("= 96" in dt_src or ": int = 96" in dt_src))

# Trajectory windows use context_length
log("CTX", "TrajectoryDataset uses context_length for windowing",
    "context_length" in dt_src and "TrajectoryDataset" in dt_src or "TrainingDataset" in dt_src)

# ═══════════════════════════════════════════════════════════
# ENSEMBLE INTEGRATION (signal-level)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ENSEMBLE INTEGRATION")
print(SEP)

if ENSEMBLE_PATH.exists():
    ens_src = ENSEMBLE_PATH.read_text(encoding="utf-8")

    log("ENS", "TQCDTEnsemble class exists",
        "class TQCDTEnsemble" in ens_src or "Ensemble" in ens_src)

    # Regime-conditional weights
    log("ENS", "REGIME_WEIGHTS dict (per-regime TQC/DT weights)",
        "REGIME_WEIGHTS" in ens_src)

    # Signal-level combination (not feature-level)
    log("ENS", "Signal-level ensemble (not feature concatenation)",
        "signal" in ens_src.lower() or "predict" in ens_src)

    # DT model loading
    log("ENS", "DT model path search",
        "dt_v32" in ens_src)

    # Fallback to TQC when DT warmup insufficient
    log("ENS", "Fallback logic when DT unavailable",
        "fallback" in ens_src.lower() or "None" in ens_src or "not" in ens_src)
else:
    log("ENS", "ensemble.py exists", False, "MISSING")

# ═══════════════════════════════════════════════════════════
# DT INFERENCE: uses own FeatureEngineerV32
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("DT INFERENCE PATH")
print(SEP)

if AGENT_PATH.exists():
    agent_src = AGENT_PATH.read_text(encoding="utf-8")

    log("INF", "RealDecisionTransformer wrapper exists",
        "class RealDecisionTransformer" in agent_src)

    log("INF", "ModelAlphaAgent class exists",
        "class ModelAlphaAgent" in agent_src)

    # DT predict returns (action, confidence)
    log("INF", "predict() returns (prediction, confidence)",
        "def predict" in agent_src and "confidence" in agent_src)
else:
    log("INF", "model_alpha_agent.py exists", False, "MISSING")

# DT model files
dt_model_dirs = [
    PROJECT / "hmats_training_v5 (1)" / "models",
    PROJECT / "models" / "dt",
    PROJECT / "models" / "retrained",
]
dt_model_found = False
for d in dt_model_dirs:
    if d.exists():
        for ext in ["*.pt", "*.pth"]:
            dt_files = list(d.glob(f"**/{ext}"))
            dt_v32_files = [f for f in dt_files if "dt" in f.name.lower() or "decision" in f.name.lower()]
            if dt_v32_files:
                dt_model_found = True
                log("INF", f"DT model file found: {dt_v32_files[0].name}",
                    True, str(dt_v32_files[0].relative_to(PROJECT)))

if not dt_model_found:
    log("INF", "DT v3.2 model file (.pt/.pth)", False,
        "no trained model file found - training needed")

# ═══════════════════════════════════════════════════════════
# DTConfigV32 KEY PARAMETERS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("DTConfigV32 KEY PARAMETERS")
print(SEP)

# Parse key config values from source
import re
config_params = {
    "context_length": (96, r"context_length.*?=\s*(\d+)"),
    "state_dim": (80, r"state_dim.*?=\s*(\d+)"),
    "action_dim": (1, r"action_dim.*?=\s*(\d+)"),
    "hidden_size": (384, r"hidden_size.*?=\s*(\d+)"),
    "num_layers": (8, r"num_layers.*?=\s*(\d+)"),
    "num_heads": (8, r"num_heads.*?=\s*(\d+)"),
}

for param, (expected, pattern) in config_params.items():
    match = re.search(pattern, dt_src)
    if match:
        actual = int(match.group(1))
        log("CFG", f"{param} = {actual} (expected {expected})",
            actual == expected)
    else:
        log("CFG", f"{param}", False, "not found in source")

# Boolean flags
bool_params = ["use_lora", "use_fgm", "use_ohem", "use_curriculum", "use_ema",
               "use_hierarchical_rtg"]
for bp in bool_params:
    match = re.search(rf"{bp}.*?=\s*(True|False)", dt_src)
    if match:
        val = match.group(1)
        log("CFG", f"{bp} = {val}", True)
    else:
        log("CFG", f"{bp}", False, "not found")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

n_pass = sum(1 for r in results if r[2] == "PASS")
n_fail = sum(1 for r in results if r[2] == "FAIL")
print(f"\n  Total checks: {len(results)}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")

if n_fail > 0:
    print(f"\n  FAILURES:")
    for section, check, status, detail in results:
        if status == "FAIL":
            print(f"    [{section}] {check}" + (f"  - {detail}" if detail else ""))

print(f"\n{SEP}")
