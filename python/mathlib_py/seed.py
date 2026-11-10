"""Seed = phrase + script.

- phrase: loaded from data/seed_phrase.txt (contextual seed string)
- script: deterministic hash/materialization pipeline over the phrase using mathlib
  (times-plank/hash, category/qbit/conjecture, event-loop/tree, relativity resonance,
  CSS Rule C tick+memory equality) to produce a reproducible seed artifact.

Usage:
  $env:PYTHONPATH='F:\\timeline\\python'; python -m mathlib_py.seed
  python -m mathlib_py.seed --phrase-file data/seed_phrase.txt --base 8 --out data/seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import (
    CalcEvent,
    HashBaseStore,
    MathOp,
    MathTree,
    base,
    group_outputs_by_category,
    process_math_event_loop,
    run_series_op,
    speculative_matrix_from_grid,
    sort_by_category,
)
from .ascii_vid import CSSRuleCConfig, quantize_tick, solve_trajectory_tick
from .aspen import BucketList, DomainStructure, derive_base_aspen_key, aspen_for_status, calculus_status, recall_aspen_in_domain
from .relativity_correction import C_CONST, correct_via_event_loop


def phrase_to_series(phrase: str, base_value: int = 8, count: int = 12) -> List[int]:
    """Derive a deterministic numeric series (xs) from the seed phrase."""
    h = hashlib.blake2b(phrase.encode("utf-8"), digest_size=32).digest()
    xs: List[int] = []
    for i in range(count):
        b = h[i % len(h)]
        # map to 1..max(1, base_value*1000) style timeline values, include 1 and scaled anchors
        v = 1 + (b ^ (i * 31)) % max(1, base_value * 125)
        xs.append(int(v))
    # ensure anchors for timeline feel
    if 1 not in xs:
        xs[0] = 1
    return xs


def run_seed(phrase: str, base_value: int = 8, out_path: Optional[Path] = None) -> Dict[str, Any]:
    phrase_hash = hashlib.blake2b(phrase.encode("utf-8"), digest_size=16).hexdigest()
    xs = phrase_to_series(phrase, base_value=base_value, count=12)
    store = HashBaseStore(base_num=base_value)

    # script: run times-plank mul+div series and collect hash-addressed structure
    mul_run = run_series_op(MathOp.MUL, base_value, xs, store=store)
    div_xs = [x for x in xs if x != 0][:8] or [1]
    div_run = run_series_op(MathOp.DIV, base_value * max(xs), div_xs)

    mul_outs = [c for r in mul_run.grid.rows for c in r.cells]
    cats = group_outputs_by_category(mul_outs)

    # derive base aspen key (phrase+script material) and compute per-loop percentage aspen
    seed_hash = mul_run.grid.rows[0].base_hash if mul_run.grid.rows else 0
    base_aspen = derive_base_aspen_key(phrase_hash, seed_hash=str(seed_hash), tree_nodes=0, series=xs)

    # event loop / tree (percentage-based aspen per iteration by calculus status)
    tree = MathTree()
    events = [CalcEvent(MathOp.MUL, base_value, x) for x in xs[:6]] + [CalcEvent(MathOp.DIV, base_value * (xs[0] or 1), d) for d in div_xs[:4]]
    process_math_event_loop(tree, events, base_aspen=base_aspen)

    # if no aspen_log populated (older behavior), compute it
    if base_aspen and not tree.aspen_log and events:
        for i, ev in enumerate(events):
            st = calculus_status("loop", i, len(events), op=ev.op.value, a=ev.a, b=ev.b)
            tree.aspen_log.append({"percentage": st["percentage_100"], "status": st, "aspen_key": aspen_for_status(base_aspen, st)})

    # bucket list spanning ticks: add aspen key to open buckets per loop (by percentage/span tick)
    bucket_list = BucketList.from_span(base_aspen or "", n=max(1, len(events)), tick_min=0.0, tick_max=1.0)
    trajectory_ticks = []
    for i, ev in enumerate(events):
        st = calculus_status("loop", i, len(events), op=ev.op.value, a=ev.a, b=ev.b)
        pct = float(st.get("percentage", 0.0))
        tt = solve_trajectory_tick(1.0, i, total_frames=len(events))
        trajectory_ticks.append(tt)
        akey = aspen_for_status(base_aspen or phrase_hash, st) if (base_aspen or phrase_hash) else ""
        if akey:
            bucket_list.add_aspen_to_open(akey, pct, tick=tt, status=st)

    # authority in domain structure: recall per-loop aspen -> field-theory computable percentage
    domain_structure = DomainStructure()
    aspen_domain_recalls = []
    if base_aspen:
        for i, ev in enumerate(events):
            st = calculus_status("loop", i, len(events), op=ev.op.value, a=ev.a, b=ev.b)
            rec = recall_aspen_in_domain(domain_structure, base_aspen, "calculus", st)
            aspen_domain_recalls.append(rec)

    # speculative/category structure
    sm = speculative_matrix_from_grid(mul_run.grid, category="seed:timeline")
    ranked = sort_by_category(sm, "seed", None)

    # relativity resonance correction sample (Einstein playbook base plate)
    rel = correct_via_event_loop(base_value, xs[0] if xs else 1, MathOp.MUL, velocity=0.5 * C_CONST)

    # CSS Rule C tick + memory-equality sample on envisioned numeric buffer
    tick = 1.0
    envisioned = [float(x) for x in xs]
    ticked = [quantize_tick(v, tick) for v in envisioned]
    css_mem_equal = (len(ticked) == len(envisioned))

    payload: Dict[str, Any] = {
        "seed_type": "phrase+script",
        "phrase": phrase,
        "phrase_hash": phrase_hash,
        "aspen_key": base_aspen,
        "base_value": base_value,
        "series_xs": xs,
        "generated_at_ns": time.time_ns(),
        "script": {
            "mul_rows": len(mul_run.grid.rows),
            "div_rows": len(div_run.grid.rows),
            "outputs_total": len(mul_outs),
            "categories": {k: len(v) for k, v in cats.items()},
            "tree_nodes": len(tree.nodes),
            "event_log_tail": tree.event_log[-6:],
            "aspen_by_percentage": tree.aspen_log,
            "aspen_domain_recalls": aspen_domain_recalls,
            "bucket_list_span_ticks": bucket_list.to_dict(),
            "trajectory_ticks": trajectory_ticks,
            "domain_structure": domain_structure.to_dict(),
            "speculative_ranked": len(ranked),
            "hash_store": store.summary(),
        },
        "relativity_resonance": {
            "base_c": C_CONST,
            "correction_sample": rel,
        },
        "css_rule_c": {
            "tick": tick,
            "memory_allocation_equal_to_input": css_mem_equal,
            "input_len": len(envisioned),
            "output_len": len(ticked),
            "sample_ticked": ticked[:8],
        },
        "note": "Seed is a phrase (semantic/context) and a script (deterministic hash/math materialization).",
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["stored_path"] = str(out_path)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate mathlib seed from phrase + script")
    p.add_argument("--phrase-file", default="data/seed_phrase.txt")
    p.add_argument("--phrase", default=None)
    p.add_argument("--base", type=int, default=8)
    p.add_argument("--out", default="data/seed.json")
    args = p.parse_args(argv)
    here = Path(__file__).resolve().parents[2]
    if args.phrase is not None:
        phrase = args.phrase
    else:
        pf = Path(args.phrase_file)
        if not pf.is_absolute():
            pf = here / pf
        phrase = pf.read_text(encoding="utf-8").strip() if pf.exists() else "mathlib timeline seed"
    out = Path(args.out)
    if not out.is_absolute():
        out = here / out
    res = run_seed(phrase, base_value=args.base, out_path=out)
    # print concise confirmation
    print(json.dumps({"seed_type": res["seed_type"], "phrase_hash": res["phrase_hash"], "stored_path": res.get("stored_path"), "categories": res["script"]["categories"], "tree_nodes": res["script"]["tree_nodes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
