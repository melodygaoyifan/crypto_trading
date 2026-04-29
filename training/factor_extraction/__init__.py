"""HMATS v5.1 Phase 6 — ML factor extraction (constitutional override).

================================================================================
[CONSTITUTIONAL OVERRIDE — Iron Law 3]
================================================================================
training/ touch approved by operator 2026-04-29 ([PARAMETER 5] unblocked
in same-session autonomous-execution authorization).

Justification: ML factor extraction for v5.1 Phase 6 — autoencoder over
the 122-feature manifest to extract a small set of latent factors with
target IC > 0.05. Goal is a new MLFactorFusionAgent at ADVISE authority
(does NOT compete with quant DECIDE per Iron Law 6).

Scope: training/factor_extraction/ NEW subdir ONLY.
       NO modification to training/drl/* (DRL ACTIVE preserved per IL5)
       NO modification to training/gmm/* (regime classifier preserved)
       NO modification to training/promotion/* or training/scripts/* etc.

Reversible: Y — delete this subdir + the new agents/ml_factor_fusion_agent.py
            shadow registration to fully revert.

Iron Laws preserved: 1 (obs_dim=126 unchanged at runtime), 5 (DRL ACTIVE
not affected), 6 (>=3 active strategies maintained — ML factor enters as
ADVISE shadow, doesn't displace quant). Iron Law 3 explicit override
recorded above.
================================================================================
"""
