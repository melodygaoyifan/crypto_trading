"""[P423] The registry of STANDING OPERATOR DECISIONS — with deadlines.

WHY: every instrument in this repo ends a verdict with "operator decision"
and nothing owns what happens next. The seat controller's "flat wins" (P421e),
the P271 failure-of-everything branch (P421), the skew re-arm question, the
jump-regime cutover — each sat in a P-entry with no owner, no deadline and no
default, which is how the 19 September reads went unread for a month (P421
mitigation (a): an OVERDUE item should be its own alarm, not a bigger table).

This module is the one place such decisions live, and two consumers make them
loud: `scripts/decision_check.py` (weekly cron, exit 3 when anything is
overdue) and the 4H Discord heartbeat ("Decisions pending" field).

DEFAULT SEMANTICS, decided here so no entry has to re-argue it: when a
deadline passes with no decision the system KEEPS ITS CURRENT STATE AND
ESCALATES — it never auto-de-risks and never auto-promotes. P141 reserves
every live config change for a human; the halts, fuse, caps and stops bound
real losses regardless; and a default that flattens the book on a missed
calendar date would make the registry a liquidation trigger. The escalation
is the mechanism: an overdue decision appears in every heartbeat until it is
recorded as decided.

Schema (`pending_decisions.json`): a list of
  {id, title, opened, deadline, owner, status: "pending"|"decided",
   evidence: [..], options: [..], default, decided_on?, decision?}
Dates are ISO YYYY-MM-DD (UTC). Decide one by setting status="decided",
decided_on and decision — in the SAME commit as the config change it names.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_PATH = Path(__file__).resolve().parent / "pending_decisions.json"

DEFAULT_ON_NO_DECISION = ("keep the current live state; escalate every heartbeat "
                          "until decided (never auto-de-risk, never auto-promote — P141)")


class RegistryUnreadable(RuntimeError):
    """The registry could not be read or parsed. Distinct from 'nothing pending'
    (P2/P199): a consumer must never render an unreadable registry as 'none'."""


def _parse_date(s: str) -> date:
    return datetime.strptime(str(s), "%Y-%m-%d").date()


def load_registry(path: Optional[Path] = None) -> List[Dict]:
    p = Path(path) if path else REGISTRY_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise RegistryUnreadable(f"{p}: {type(e).__name__}: {e}") from e
    if not isinstance(data, list):
        raise RegistryUnreadable(f"{p}: expected a list of decisions")
    out = []
    for d in data:
        if not isinstance(d, dict) or "id" not in d or "deadline" not in d:
            raise RegistryUnreadable(f"{p}: malformed entry {d!r}")
        _parse_date(d["deadline"])  # validate
        out.append(d)
    return out


def pending(registry: List[Dict]) -> List[Dict]:
    return [d for d in registry if str(d.get("status", "pending")).lower() == "pending"]


def overdue(registry: List[Dict], today: Optional[date] = None) -> List[Dict]:
    today = today or datetime.now(timezone.utc).date()
    return [d for d in pending(registry) if _parse_date(d["deadline"]) < today]


def days_overdue(d: Dict, today: Optional[date] = None) -> int:
    today = today or datetime.now(timezone.utc).date()
    return (today - _parse_date(d["deadline"])).days


def heartbeat_line(path: Optional[Path] = None, today: Optional[date] = None) -> str:
    """One line for the 4H heartbeat. Raises RegistryUnreadable rather than
    returning 'none pending' on a broken file."""
    today = today or datetime.now(timezone.utc).date()
    reg = load_registry(path)
    pend = pending(reg)
    if not pend:
        return "none pending"
    late = overdue(reg, today)
    if not late:
        nxt = min(pend, key=lambda d: _parse_date(d["deadline"]))
        return f"{len(pend)} pending (next due {nxt['deadline']}: {nxt['id']})"
    worst = max(late, key=lambda d: days_overdue(d, today))
    names = ", ".join(f"{d['id']} (+{days_overdue(d, today)}d)" for d in late)
    return (f"{len(pend)} pending, {len(late)} OVERDUE: {names} — default: keep "
            f"state + escalate; decide via analytics/decisions/pending_decisions.json "
            f"(oldest {worst['id']})")
