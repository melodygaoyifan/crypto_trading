from core.runtime_authority import (
    ALLOWED_RUNTIME_AUTHORITY_STATES,
    build_runtime_authority_snapshot,
    normalize_runtime_authority,
)
from core.runtime_export_service import (
    build_exit_alpha_review_summary,
    inject_asset_authority_states,
)


def test_normalize_runtime_authority_maps_existing_runtime_labels():
    assert normalize_runtime_authority("composite_toxicity", authority="EXECUTION") == "ADVISORY"
    assert normalize_runtime_authority("lead_lag", authority="CONTEXT") == "ADVISORY"
    assert normalize_runtime_authority("g6", authority="SIZING", mode="LIVE") == "BOUNDED_LIVE"
    assert normalize_runtime_authority("aggressive_allocator", authority="SIZING_DEFERRED") == "SHADOW"
    assert normalize_runtime_authority("learned_execution_policy", effective_mode="ADVISORY_HEURISTIC") == "ADVISORY"
    assert normalize_runtime_authority("drl", gate_level="SHADOW", execution_authority="OFF") == "SHADOW"
    assert normalize_runtime_authority("drl", gate_level="EXIT_ONLY", execution_authority="ON") == "BOUNDED_LIVE"


def test_build_runtime_authority_snapshot_uses_only_allowed_states():
    snap = build_runtime_authority_snapshot(
        composite_toxicity_authority="EXECUTION",
        learned_exec_authority="EXECUTION",
        learned_exec_effective_mode="ADVISORY_HEURISTIC",
        g6_authority="SIZING",
        g6_mode="LIVE",
        aggressive_allocator_authority="SIZING_DEFERRED",
        lead_lag_authority="CONTEXT",
        model_alpha_authority="CONTEXT",
        drl_gate_level="SHADOW",
        drl_execution_authority="OFF",
    )

    allowed = set(ALLOWED_RUNTIME_AUTHORITY_STATES)
    assert set(snap["allowed_states"]) == allowed
    assert set(snap["by_module"]).issuperset(
        {
            "composite_toxicity",
            "learned_execution_policy",
            "g6",
            "aggressive_allocator",
            "lead_lag",
            "model_alpha",
            "drl",
        }
    )
    assert set(snap["by_module"].values()).issubset(allowed)


def test_inject_asset_authority_states_overrides_stale_snapshot_values():
    enriched = inject_asset_authority_states(
        {
            "BTC": {
                "authority_states": {"g6": "DISABLED"},
                "g6_authority_state": "DISABLED",
                "learned_exec_authority_state": "DISABLED",
            }
        },
        {
            "by_asset": {
                "BTC": {
                    "g6": "BOUNDED_LIVE",
                    "learned_execution_policy": "ADVISORY",
                }
            }
        },
    )

    assert enriched["BTC"]["authority_states"]["g6"] == "BOUNDED_LIVE"
    assert enriched["BTC"]["g6_authority_state"] == "BOUNDED_LIVE"
    assert enriched["BTC"]["learned_exec_authority_state"] == "ADVISORY"


def test_build_exit_alpha_review_summary_marks_failed_tracker_unavailable():
    class BadTracker:
        def summary_dict(self):
            raise RuntimeError("boom")

    summary = build_exit_alpha_review_summary(BadTracker())

    assert summary["available"] is False
    assert summary["tracked_trades"] == 0
