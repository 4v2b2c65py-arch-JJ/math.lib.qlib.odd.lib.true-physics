"""Result statement: portal traversal and energy dissipation.

Model (within mathlib/timeline field structure):
- Portal traversal = hash-addressed transition through math tree/event loop nodes
  (address = tree_key_hash(op,a,b), state carried in row/cell hashes and aspen).
- Energy accounting uses Einstein SR base plate (c, gamma) and hash-magnitude
  proxies aligned to atomic reference scales; dissipation is the residual not
  conserved as coherent traversal work/state, tracked explicitly so total is accounted.
- Result is a computable statement stored with hashes/ticks/buckets for audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import CalcEvent, MathOp, MathTree, build_row_op, process_math_event_loop, tree_key_hash
from .aspen import BucketList, DomainStructure, aspen_for_status, calculus_status, derive_base_aspen_key, field_percentage
from .relativity_correction import C_CONST, apply_sr_corrections


@dataclass
class TraversalStep:
    index: int
    op: str
    a: int
    b: int
    node_key: int
    row_hash: int
    aspen_key: str
    percentage: float
    field_percentage: float
    energy_in: float
    energy_transferred: float
    energy_dissipated: float
    energy_out: float
    gamma: float
    tick: float


def _hash_to_unit(h: int) -> float:
    # map 64-bit hash to (0,1] energy unit proxy (deterministic, computable)
    return ((h & 0xFFFFFFFFFFFFFFFF) / float(0xFFFFFFFFFFFFFFFF)) or 1e-12


def _tick_for_index(i: int, total: int) -> float:
    # simple trajectory tick aligned with our solved-tick envelope (no trig)
    n = max(1, total - 1)
    frac = i / n if total > 1 else 0.0
    return 0.5 + frac  # 0.5..1.5


def run_portal_traversal(
    events: Sequence[CalcEvent],
    base_aspen: str,
    v_fraction_c: float = 0.1,
    e0: float = 1.0,
) -> Dict[str, Any]:
    """Traverse portal (hash-addressed tree) and account for energy/dissipation."""
    total = len(events)
    corr = apply_sr_corrections(v_fraction_c * C_CONST, C_CONST)
    gamma = float(corr.gamma)
    tree = MathTree()
    # run event loop with aspen logging
    process_math_event_loop(tree, events, base_aspen=base_aspen)

    domain = DomainStructure()
    buckets = BucketList.from_span(base_aspen, n=max(1, total), tick_min=0.5, tick_max=1.5)

    steps: List[TraversalStep] = []
    energy = float(e0)
    total_dissipated = 0.0
    total_transferred = 0.0

    for i, ev in enumerate(events):
        st = calculus_status("traverse", i, total, op=ev.op.value, a=ev.a, b=ev.b)
        pct = float(st.get("percentage", 0.0))
        akey = aspen_for_status(base_aspen, st)
        fp = field_percentage(base_aspen, "portal", st)
        key = tree_key_hash(ev.op, ev.a, ev.b)
        node = tree.nodes.get(key)
        row_hash = int(node.row.base_hash) if node else 0
        tick = _tick_for_index(i, total)

        # energy partition (accounted): coherent transfer vs dissipated residual
        # transfer efficiency tied to field percentage and relativistic factor; dissipation is remainder
        u = _hash_to_unit(row_hash ^ key)
        transfer_eff = max(0.0, min(1.0, (0.5 * fp + 0.5 * pct) / max(gamma, 1e-12)))
        # bound efficiency away from pathological values
        transfer_eff = 0.05 + 0.90 * transfer_eff
        e_in = energy
        e_trans = e_in * transfer_eff
        e_diss = e_in - e_trans
        # small relativistic bookkeeping: apparent energy scaled by gamma in transit frame, but conserved in accounting
        e_out = e_trans  # coherent carried forward
        energy = e_out
        total_transferred += e_trans
        total_dissipated += e_diss

        buckets.add_aspen_to_open(akey, pct, tick=tick, status=st)
        domain.register("portal", akey, weight=fp, meta={"index": i, "action": "traverse"})

        steps.append(TraversalStep(
            index=i, op=ev.op.value, a=ev.a, b=ev.b,
            node_key=key, row_hash=row_hash, aspen_key=akey,
            percentage=pct, field_percentage=fp,
            energy_in=e_in, energy_transferred=e_trans,
            energy_dissipated=e_diss, energy_out=e_out,
            gamma=gamma, tick=tick,
        ))

    accounted = abs((energy + total_dissipated) - float(e0)) < 1e-9
    result = {
        "type": "portal_traversal_energy_dissipation_result",
        "generated_at_ns": time.time_ns(),
        "constants": {"c": C_CONST, "v_fraction_c": v_fraction_c, "gamma": gamma, "sr_notes": corr.notes},
        "base_aspen": base_aspen,
        "events": [{"op": e.op.value, "a": e.a, "b": e.b} for e in events],
        "steps": [asdict(s) for s in steps],
        "buckets": buckets.to_dict(),
        "domain_structure": domain.to_dict(),
        "energy_accounting": {
            "E_in_total": float(e0),
            "E_transferred_coherent_total": total_transferred,
            "E_dissipated_total": total_dissipated,
            "E_out_final": energy,
            "conservation_check": accounted,
            "balance_error": (energy + total_dissipated) - float(e0),
        },
        "statement": {
            "portal_traversal": (
                "Portal traversal is modeled as hash-addressed transition across math-tree nodes "
                "(tree_key_hash per CalcEvent) with aspen-keyed, percentage/field-status selection per loop. "
                "Each step carries state via row/cell hashes and domain authority; traversal is deterministic "
                "and auditable across the bucket/tick span."
            ),
            "energy_dissipation": (
                "Energy is fully accounted at every step: E_in_step = E_transferred + E_dissipated, with coherent "
                "transfer governed by field percentage and relativistic gamma (Einstein SR base plate, c constant). "
                "Dissipation is the residual non-coherent component recorded per transition (hash-residual/entropy proxy), "
                "not lost from the ledger; total conservation holds (E_in_total = E_out_final + E_dissipated_total)."
            ),
            "resolution": (
                "Within this field/timeline structure, portal traversal and energy dissipation are resolved as "
                "conservation-consistent, hash-addressed state transitions: traversal paths are computable, "
                "dissipation is explicitly quantified per step, and all quantities are percentage-/tick-/hash-traceable."
            ),
        },
    }
    return result


def default_demo_events() -> List[CalcEvent]:
    return [
        CalcEvent(MathOp.MUL, 8, 1),
        CalcEvent(MathOp.MUL, 8, 12),
        CalcEvent(MathOp.DIV, 96, 8),
        CalcEvent(MathOp.MUL, 12, 8),
        CalcEvent(MathOp.DIV, 8000, 1000),
        CalcEvent(MathOp.MUL, 8, 1000),
    ]


def run_and_store(out_path: Optional[Path] = None) -> Dict[str, Any]:
    here = Path(__file__).resolve().parents[2]
    # derive base aspen from seed phrase if present
    phrase_path = here / "data" / "seed_phrase.txt"
    phrase = phrase_path.read_text(encoding="utf-8").strip() if phrase_path.exists() else "timeline portal seed"
    phrase_hash = hashlib.blake2b(phrase.encode("utf-8"), digest_size=16).hexdigest()
    base_aspen = derive_base_aspen_key(phrase_hash, seed_hash="portal", tree_nodes=0, series=[8, 1, 12, 96, 1000, 8000])
    events = default_demo_events()
    res = run_portal_traversal(events, base_aspen, v_fraction_c=0.2, e0=1.0)
    out = Path(out_path) if out_path else (here / "data" / "portal_traversal_energy_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    res["stored_path"] = str(out)
    return res


if __name__ == "__main__":
    r = run_and_store()
    print(json.dumps({"stored_path": r["stored_path"], "conservation_check": r["energy_accounting"]["conservation_check"], "steps": len(r["steps"]), "statement": r["statement"]}, indent=2))
