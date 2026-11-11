"""Lore role basis decoding.

Decodes hash-addressed outputs/qbits and contextual analysis (rings/strings/stings
in depth) into lore/role/basis representations for new contextual analysis.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .shortcuts import simplify_log, shorten_hash


def _safe_log_basis(value: Any, base: float = 10.0) -> Optional[Dict[str, Any]]:
    try:
        vf = float(value)
    except Exception:
        return None
    av = abs(vf)
    if av == 0.0:
        return {"x": vf, "zero": True, "basis": [0.0, 0.0], "sign": 0}
    try:
        s = simplify_log(av, base=base)
        exp = float(s.get("exp10", 0.0) if base == 10.0 else 0.0)
        mant = float(s.get("mantissa", av))
        basis = [exp, math.log10(mant) if (base == 10.0 and mant > 0) else float(s.get("value", 0.0)) - exp]
        s["basis"] = basis
        s["sign"] = -1 if vf < 0 else 1
        return s
    except Exception:
        return None


def qbit_basis(qbit: Any) -> Dict[str, Any]:
    if qbit is None:
        return {"present": False}
    alpha = float(getattr(qbit, "alpha", 0.0))
    beta = float(getattr(qbit, "beta", 0.0))
    phase = float(getattr(qbit, "phase", 0.0))
    p_plus = alpha * alpha
    p_minus = beta * beta
    return {
        "present": True,
        "label": getattr(qbit, "label", "|psi>"),
        "alpha": alpha,
        "beta": beta,
        "phase": phase,
        "basis": [alpha, beta, math.sin(phase), math.cos(phase)],
        "probabilities": {"plus": p_plus, "minus": p_minus},
    }


def role_for_output(o: Any) -> Dict[str, Any]:
    agent_id = getattr(o, "agent_id", None)
    conjecture = bool(getattr(o, "conjecture", False))
    category = str(getattr(o, "category", ""))
    if conjecture:
        role = "conjecture"
    elif agent_id == 1:
        role = "reference"
    elif agent_id == 0:
        role = "primary"
    else:
        role = "hypothesis"
    subroles = []
    if "result" in category:
        subroles.append("result")
    if "agent" in category:
        subroles.append("agent")
    if getattr(o, "stored_hash_value", None) is not None:
        subroles.append("hash-stored")
    return {
        "role": role,
        "subroles": subroles,
        "agent_id": agent_id,
        "conjecture": conjecture,
        "category": category,
    }


def decode_output_lore(o: Any, base: float = 10.0) -> Dict[str, Any]:
    val = getattr(o, "value", None)
    h = getattr(o, "hash", None)
    addr = getattr(o, "address", None)
    qb = getattr(o, "qbit", None)
    lore = {
        "value": val,
        "hash": h,
        "hash_short": shorten_hash(int(h)) if h is not None else None,
        "address": {"x": getattr(addr, "x", None), "y": getattr(addr, "y", None), "z": getattr(addr, "z", None)} if addr is not None else None,
        "role": role_for_output(o),
        "qbit_basis": qbit_basis(qb),
        "value_basis": _safe_log_basis(val, base=base),
    }
    lore["lore"] = (
        f"role={lore['role']['role']} agent={lore['role']['agent_id']} "
        f"cat={lore['role']['category']} hash={lore['hash_short']} val={val}"
    )
    return lore


def decode_ring_string_sting_context(contextual_analysis: Dict[str, Any], base: float = 10.0) -> Dict[str, Any]:
    rings = contextual_analysis.get("rings") or []
    stings = contextual_analysis.get("stings") or []
    depth = contextual_analysis.get("depth") or {}
    ring_lore = []
    for r in rings:
        members = r.get("members") or []
        strings = r.get("strings") or []
        ring_role = "active-ring" if members else "empty-ring"
        phase_center = float(r.get("phase_center", 0.0))
        ring_basis = [math.cos(phase_center), math.sin(phase_center), float(len(members))]
        decoded_strings = []
        for s in strings:
            sv = s.get("value")
            decoded_strings.append({
                "idx": s.get("idx"),
                "value": sv,
                "role": "string",
                "value_basis": _safe_log_basis(sv, base=base),
                "log_expr": s.get("log_expr"),
            })
        ring_lore.append({
            "ring": r.get("ring"),
            "role": ring_role,
            "basis": ring_basis,
            "member_count": len(members),
            "strings": decoded_strings,
            "lore": f"ring={r.get('ring')} role={ring_role} members={len(members)} strings={len(decoded_strings)}",
        })
    sting_lore = []
    for s in stings:
        sv = s.get("value")
        sting_lore.append({
            "idx": s.get("idx"),
            "ring": s.get("ring"),
            "role": "sting",
            "depth_hint": s.get("depth_hint"),
            "basis": _safe_log_basis(sv, base=base),
            "lore": f"sting idx={s.get('idx')} ring={s.get('ring')} depth={s.get('depth_hint')} val={sv}",
        })
    levels = depth.get("levels") or []
    depth_lore = [{
        "depth": lv.get("depth"),
        "role": "depth-layer",
        "basis": [float(lv.get("ring_count", 0)), float(lv.get("string_count", 0)), float(lv.get("sting_count", 0))],
        "lore": f"depth={lv.get('depth')} rings={lv.get('ring_count')} strings={lv.get('string_count')} stings={lv.get('sting_count')}",
    } for lv in levels]
    return {
        "rings": ring_lore,
        "stings": sting_lore,
        "depth": {"levels": depth_lore, "max_depth": depth.get("max_depth"), "in_depth": depth.get("in_depth", True)},
        "counts": contextual_analysis.get("counts"),
    }


def decode_lore_role_basis(outputs: Optional[Sequence[Any]] = None,
                           contextual_analysis: Optional[Dict[str, Any]] = None,
                           base: float = 10.0,
                           depth: Optional[int] = None) -> Dict[str, Any]:
    out_lore = []
    outs = list(outputs or [])
    if depth is not None:
        outs = outs[: max(0, int(depth))]
    for o in outs:
        out_lore.append(decode_output_lore(o, base=base))
    ctx_lore = decode_ring_string_sting_context(contextual_analysis, base=base) if contextual_analysis else None
    roles: Dict[str, int] = {}
    for e in out_lore:
        r = (e.get("role") or {}).get("role", "unknown")
        roles[r] = roles.get(r, 0) + 1
    return {
        "type": "lore_role_basis_decoding",
        "base": base,
        "outputs": out_lore,
        "output_count": len(out_lore),
        "roles": roles,
        "contextual": ctx_lore,
    }


def decode_grid_lore_role_basis(grid: Any, base: float = 10.0, depth: Optional[int] = None) -> Dict[str, Any]:
    rows = getattr(grid, "rows", []) or []
    sel_rows = rows if depth is None else rows[: max(0, int(depth))]
    outputs = [c for r in sel_rows for c in (getattr(r, "cells", []) or [])]
    res = decode_lore_role_basis(outputs=outputs, contextual_analysis=None, base=base, depth=None)
    res["grid_depth"] = len(sel_rows)
    res["row_count"] = len(rows)
    return res
