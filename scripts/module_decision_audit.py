"""
Module-by-module decision audit for the latest paper diagnostics.

Usage:
    python -X utf8 scripts/module_decision_audit.py
    python -X utf8 scripts/module_decision_audit.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DIAG_DIR = DATA_DIR / "diagnostics"
ASSETS = ("BTC", "ETH", "SOL")


@dataclass
class ComponentStatus:
    asset: str
    name: str
    status: str
    called: bool
    consumed: bool
    applicable: bool
    expect_consumption: bool
    note: str
    output: str


@dataclass
class AssetAudit:
    asset: str
    diag_path: str
    tick_time: str
    final_intent: dict[str, Any]
    summary: dict[str, Any]
    components: list[ComponentStatus]


def load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_diag_for_asset(asset: str) -> Optional[Path]:
    candidates = sorted(
        DIAG_DIR.glob(f"diag_{asset}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def classify_component(name: str, data: dict[str, Any]) -> str:
    applicable = bool(data.get("applicable", True))
    called = bool(data.get("called", False))
    consumed = bool(data.get("consumed", False))
    expect_consumption = bool(data.get("expect_consumption", True))

    if not applicable:
        return "not_applicable"
    if not called:
        return "inactive"
    if consumed:
        return "effective"
    if expect_consumption:
        return "missed"
    return "observer"


def load_asset_audit(asset: str) -> Optional[AssetAudit]:
    path = latest_diag_for_asset(asset)
    if path is None:
        return None
    obj = load_json(path)
    if not obj:
        return None
    components: list[ComponentStatus] = []
    for name, raw in obj.get("components", {}).items():
        components.append(
            ComponentStatus(
                asset=asset,
                name=name,
                status=classify_component(name, raw),
                called=bool(raw.get("called", False)),
                consumed=bool(raw.get("consumed", False)),
                applicable=bool(raw.get("applicable", True)),
                expect_consumption=bool(raw.get("expect_consumption", True)),
                note=str(raw.get("note", "")),
                output=str(raw.get("output", "")),
            )
        )
    components.sort(key=lambda row: (row.status, row.name))
    return AssetAudit(
        asset=asset,
        diag_path=str(path.relative_to(ROOT)),
        tick_time=str(obj.get("tick_time", "")),
        final_intent=dict(obj.get("final_intent", {})),
        summary=dict(obj.get("summary", {})),
        components=components,
    )


def short_list(items: list[str], limit: int = 10) -> str:
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    head = ", ".join(items[:limit])
    return f"{head}, ... (+{len(items) - limit})"


def render_asset_report(audit: AssetAudit) -> list[str]:
    by_status: dict[str, list[str]] = defaultdict(list)
    for row in audit.components:
        by_status[row.status].append(row.name)
    for names in by_status.values():
        names.sort()

    final_intent = audit.final_intent
    summary = audit.summary

    lines = [
        f"[{audit.asset}] {audit.tick_time}",
        f"  diag: {audit.diag_path}",
        (
            "  intent: "
            f"dir={float(final_intent.get('direction', 0.0)):+.2f} "
            f"exp={float(final_intent.get('target_exposure', 0.0)):.1%} "
            f"alpha_pass={bool(final_intent.get('alpha_gate_passed', False))} "
            f"veto={bool(final_intent.get('veto_active', False))} "
            f"reason={final_intent.get('veto_reason', 'n/a')}"
        ),
        (
            "  summary: "
            f"called={summary.get('called', 0)}/{summary.get('total_probes', 0)} "
            f"consumed={summary.get('consumed', 0)} "
            f"coverage={summary.get('coverage_pct', 0):.1f}%"
        ),
        f"  effective: {short_list(by_status.get('effective', []))}",
        f"  observer: {short_list(by_status.get('observer', []))}",
        f"  missed: {short_list(by_status.get('missed', []))}",
        f"  inactive: {short_list(by_status.get('inactive', []))}",
        f"  not_applicable: {short_list(by_status.get('not_applicable', []))}",
    ]
    return lines


def aggregate(audits: list[AssetAudit]) -> dict[str, Any]:
    per_module: dict[str, dict[str, str]] = defaultdict(dict)
    status_counts: Counter[str] = Counter()
    for audit in audits:
        for row in audit.components:
            per_module[row.name][audit.asset] = row.status
            status_counts[row.status] += 1

    always_effective: list[str] = []
    always_observer: list[str] = []
    always_missed: list[str] = []
    always_inactive: list[str] = []
    always_not_applicable: list[str] = []
    mixed: dict[str, dict[str, str]] = {}

    for name, statuses in sorted(per_module.items()):
        values = {statuses.get(asset, "missing") for asset in ASSETS}
        if values == {"effective"}:
            always_effective.append(name)
        elif values == {"observer"}:
            always_observer.append(name)
        elif values == {"missed"}:
            always_missed.append(name)
        elif values == {"inactive"}:
            always_inactive.append(name)
        elif values == {"not_applicable"}:
            always_not_applicable.append(name)
        else:
            mixed[name] = statuses

    return {
        "status_counts": dict(status_counts),
        "always_effective": always_effective,
        "always_observer": always_observer,
        "always_missed": always_missed,
        "always_inactive": always_inactive,
        "always_not_applicable": always_not_applicable,
        "mixed": mixed,
    }


def render_text(audits: list[AssetAudit], summary: dict[str, Any]) -> str:
    lines = [
        "Latest decision-module audit",
        "",
        "Status legend:",
        "  effective     module output changed this tick's decision path",
        "  observer      module ran, but correctly stayed observational/idle",
        "  missed        module ran and was expected to matter, but did not",
        "  inactive      module was applicable but not called this tick",
        "  not_applicable module does not apply to this asset/tick",
        "",
        "Global summary:",
        f"  status_counts: {summary['status_counts']}",
        f"  always_effective: {short_list(summary['always_effective'], limit=20)}",
        f"  always_observer: {short_list(summary['always_observer'], limit=20)}",
        f"  always_missed: {short_list(summary['always_missed'], limit=20)}",
        f"  always_inactive: {short_list(summary['always_inactive'], limit=20)}",
        f"  always_not_applicable: {short_list(summary['always_not_applicable'], limit=20)}",
    ]
    if summary["mixed"]:
        lines.append("  mixed:")
        for name, statuses in sorted(summary["mixed"].items()):
            rendered = ", ".join(f"{asset}={statuses.get(asset, 'missing')}" for asset in ASSETS)
            lines.append(f"    {name}: {rendered}")

    for audit in audits:
        lines.extend(["", *render_asset_report(audit)])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether modules reached the latest final decision.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    audits = [audit for asset in ASSETS if (audit := load_asset_audit(asset)) is not None]
    if not audits:
        print("No diagnostics found under data/diagnostics.", file=sys.stderr)
        return 1

    summary = aggregate(audits)
    payload = {
        "assets": [
            {
                "asset": audit.asset,
                "diag_path": audit.diag_path,
                "tick_time": audit.tick_time,
                "final_intent": audit.final_intent,
                "summary": audit.summary,
                "components": [asdict(component) for component in audit.components],
            }
            for audit in audits
        ],
        "aggregate": summary,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(audits, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
