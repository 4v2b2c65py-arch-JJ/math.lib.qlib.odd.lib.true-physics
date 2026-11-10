"""Field inspect run: align calculation values with atomic/physical constants (CODATA).

Runs inspection for every calculation (mul/div rows) and produces an alignment report
mapping output/hash values to corrected atomic reference scales.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import MathOp, TimesPlankGrid, build_grid_ops, group_outputs_by_category


DEFAULT_ATOMIC_KEYS = [
    "speed of light in vacuum",
    "Planck constant",
    "reduced Planck constant",
    "elementary charge",
    "electron mass",
    "proton mass",
    "neutron mass",
    "fine-structure constant",
    "Rydberg constant",
    "atomic mass constant",
    "standard acceleration of gravity",
]


_ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
         "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
_SCALES = ["","thousand","million","billion","trillion"]


def int_to_words(n: int) -> str:
    """Convert integer to English words for word-logged ingress values."""
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n == 0:
        return "zero"
    neg = n < 0
    n = abs(n)
    parts: List[str] = []
    scale_idx = 0
    while n > 0:
        chunk = n % 1000
        if chunk:
            chunk_words = _chunk_to_words(chunk)
            if _SCALES[scale_idx]:
                chunk_words += " " + _SCALES[scale_idx]
            parts.append(chunk_words)
        n //= 1000
        scale_idx += 1
        if scale_idx >= len(_SCALES) and n > 0:
            parts.append(str(n))  # fallback for very large
            break
    words = " ".join(reversed(parts)).strip()
    return ("negative " + words) if neg else words


def _chunk_to_words(n: int) -> str:
    words = []
    if n >= 100:
        words.append(_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        w = _TENS[n // 10]
        if n % 10:
            w += " " + _ONES[n % 10]
        words.append(w)
    elif n > 0:
        words.append(_ONES[n])
    return " ".join(words)


def word_log_integers(values: Sequence[int]) -> Dict[str, Any]:
    """Word-log ingressed integers and their adjacents."""
    entries = []
    for v in values:
        try:
            iv = int(v)
        except Exception:
            iv = 0
        adj = [iv - 1, iv, iv + 1]
        entries.append({
            "value": iv,
            "value_words": int_to_words(iv),
            "adjacents": adj,
            "adjacents_words": [int_to_words(x) for x in adj],
        })
    return {"entries": entries}


@dataclass
class AlignmentRecord:
    op: str
    a: int
    b: int
    row_base_hash: int
    cell_index: int
    agent_id: int
    category: str
    value: float
    stored_hash: int
    aligned_constant: str
    aligned_constant_value: float
    aligned_unit: str
    scale_ratio: float
    residual: float
    qbit: Optional[Dict[str, float]] = None
    value_words: str = ""
    ingress_word_log: Optional[Dict[str, Any]] = None


@dataclass
class FieldInspectReport:
    generated_at_ns: int
    atomic_dataset: str
    total_calculations: int
    total_outputs: int
    alignments: List[AlignmentRecord]
    summary: Dict[str, Any]


def _load_atomic_records(path: Optional[Path] = None) -> Tuple[Dict[str, Any], str]:
    # default to data/physics_codata_corrected.json relative to project root
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "physics_codata_corrected.json"
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        recs = data.get("records", data)
        return recs, str(p)
    # fallback minimal set (SI) if dataset missing
    recs = {
        "speed of light in vacuum": {"key": "speed of light in vacuum", "value": 299792458.0, "unit": "m s^-1"},
        "Planck constant": {"key": "Planck constant", "value": 6.62607015e-34, "unit": "J Hz^-1"},
        "reduced Planck constant": {"key": "reduced Planck constant", "value": 1.0545718176461565e-34, "unit": "J s"},
        "elementary charge": {"key": "elementary charge", "value": 1.602176634e-19, "unit": "C"},
        "electron mass": {"key": "electron mass", "value": 9.1093837015e-31, "unit": "kg"},
        "proton mass": {"key": "proton mass", "value": 1.67262192369e-27, "unit": "kg"},
        "fine-structure constant": {"key": "fine-structure constant", "value": 0.0072973525693, "unit": ""},
        "atomic mass constant": {"key": "atomic mass constant", "value": 1.6605390666e-27, "unit": "kg"},
    }
    return recs, "(fallback)"


def _pick_atomic_targets(recs: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> List[Tuple[str, float, str]]:
    keys = list(keys) if keys is not None else DEFAULT_ATOMIC_KEYS
    out: List[Tuple[str, float, str]] = []
    for k in keys:
        r = recs.get(k)
        if not r:
            continue
        try:
            v = float(r.get("value"))
        except Exception:
            continue
        if v == 0.0 or math.isnan(v) or math.isinf(v):
            continue
        out.append((str(r.get("key", k)), v, str(r.get("unit", ""))))
    # ensure some targets exist
    if not out:
        for k, r in recs.items():
            try:
                v = float(r.get("value", 0.0))
            except Exception:
                continue
            if v != 0.0 and math.isfinite(v):
                out.append((str(k), v, str(r.get("unit", ""))))
            if len(out) >= 8:
                break
    return out


def _align_value(val: float, targets: List[Tuple[str, float, str]]) -> Tuple[str, float, str, float, float]:
    """Align a numeric value to the closest atomic constant in log-magnitude space.
    Returns (key, const_value, unit, scale_ratio=value/const, residual).
    """
    if not targets:
        return ("none", 1.0, "", float(val), 0.0)
    av = abs(float(val))
    if av == 0.0 or not math.isfinite(av):
        # map zero/invalid to fine-structure or first target deterministically
        k, cv, u = targets[0]
        return (k, cv, u, 0.0, 0.0)
    lv = math.log10(av)
    best = None
    for k, cv, u in targets:
        ac = abs(cv)
        if ac == 0.0 or not math.isfinite(ac):
            continue
        lc = math.log10(ac)
        # residual as distance in log10 space (scale-invariant alignment)
        res = lv - lc
        score = abs(res)
        if best is None or score < best[0]:
            best = (score, res, k, cv, u)
    if best is None:
        k, cv, u = targets[0]
        return (k, cv, u, av / cv, 0.0)
    _, res, k, cv, u = best
    ratio = float(val) / cv if cv != 0 else 0.0
    return (k, cv, u, ratio, res)


def inspect_grid(
    grid: TimesPlankGrid,
    calcs: Sequence[Tuple[MathOp, int, int]],
    atomic_path: Optional[Path] = None,
    keys: Optional[Sequence[str]] = None,
) -> FieldInspectReport:
    recs, src = _load_atomic_records(atomic_path)
    targets = _pick_atomic_targets(recs, keys)
    alignments: List[AlignmentRecord] = []
    total_out = 0
    for i, (row, calc) in enumerate(zip(grid.rows, calcs)):
        op, a, b = calc
        for j, cell in enumerate(row.cells):
            total_out += 1
            # align both interpreted value and hash-magnitude view; primary record on value
            val = float(cell.value)
            k, cv, u, ratio, res = _align_value(val, targets)
            q = None
            if getattr(cell, "qbit", None) is not None:
                qb = cell.qbit
                q = {"alpha": float(getattr(qb, "alpha", 0.0)), "beta": float(getattr(qb, "beta", 0.0)), "phase": float(getattr(qb, "phase", 0.0))}
            # word-log ingressed integers and adjacents (inputs a,b and output value)
            ingress_vals = [int(a), int(b)]
            try:
                out_iv = int(round(val))
            except Exception:
                out_iv = 0
            ingress_vals.append(out_iv)
            wlog = word_log_integers(ingress_vals)
            alignments.append(AlignmentRecord(
                op=op.value if hasattr(op, "value") else str(op),
                a=int(a), b=int(b),
                row_base_hash=int(row.base_hash),
                cell_index=j,
                agent_id=int(cell.agent_id),
                category=str(getattr(cell, "category", "calc")),
                value=val,
                stored_hash=int(getattr(cell, "stored_hash_value", None) or cell.hash),
                aligned_constant=k,
                aligned_constant_value=float(cv),
                aligned_unit=u,
                scale_ratio=float(ratio),
                residual=float(res),
                qbit=q,
                value_words=int_to_words(out_iv),
                ingress_word_log=wlog,
            ))

    # summary: per-constant counts and average abs residual
    by_const: Dict[str, Dict[str, Any]] = {}
    for a in alignments:
        d = by_const.setdefault(a.aligned_constant, {"count": 0, "sum_abs_residual": 0.0, "unit": a.aligned_unit, "value": a.aligned_constant_value})
        d["count"] += 1
        d["sum_abs_residual"] += abs(a.residual)
    for d in by_const.values():
        d["avg_abs_log10_residual"] = (d["sum_abs_residual"] / d["count"]) if d["count"] else 0.0
        del d["sum_abs_residual"]

    # aggregate word-log summary for ingressed integers and adjacents
    word_log_index: Dict[str, Dict[str, Any]] = {}
    for a in alignments:
        if not a.ingress_word_log:
            continue
        for e in a.ingress_word_log.get("entries", []):
            k = str(e.get("value"))
            if k not in word_log_index:
                word_log_index[k] = {
                    "value": e.get("value"),
                    "value_words": e.get("value_words"),
                    "adjacents": e.get("adjacents"),
                    "adjacents_words": e.get("adjacents_words"),
                    "count": 0,
                }
            word_log_index[k]["count"] += 1

    report = FieldInspectReport(
        generated_at_ns=time.time_ns(),
        atomic_dataset=src,
        total_calculations=len(calcs),
        total_outputs=total_out,
        alignments=alignments,
        summary={
            "targets_considered": len(targets),
            "aligned_constants": by_const,
            "categories": {k: len(v) for k, v in group_outputs_by_category([c for r in grid.rows for c in r.cells]).items()},
            "word_logged_integers": word_log_index,
        },
    )
    return report


def run_field_inspect(
    calcs: Sequence[Tuple[MathOp, int, int]],
    atomic_path: Optional[Path] = None,
    keys: Optional[Sequence[str]] = None,
) -> FieldInspectReport:
    """Run field inspection for every calculation and align outputs to atomic values."""
    grid = build_grid_ops(list(calcs))
    return inspect_grid(grid, calcs, atomic_path=atomic_path, keys=keys)


def run_series_field_inspect(
    op: MathOp,
    base_value: int,
    xs: Sequence[int],
    atomic_path: Optional[Path] = None,
    keys: Optional[Sequence[str]] = None,
) -> FieldInspectReport:
    calcs = [(op, int(base_value), int(x)) for x in xs]
    return run_field_inspect(calcs, atomic_path=atomic_path, keys=keys)


def report_to_dict(report: FieldInspectReport) -> Dict[str, Any]:
    return {
        "generated_at_ns": report.generated_at_ns,
        "atomic_dataset": report.atomic_dataset,
        "total_calculations": report.total_calculations,
        "total_outputs": report.total_outputs,
        "summary": report.summary,
        "alignments": [asdict(a) for a in report.alignments],
    }


def store_report(report: FieldInspectReport, out_path: Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_to_dict(report), indent=2, default=str), encoding="utf-8")
    return str(out_path)


def run_and_store_field_inspect(
    op: MathOp,
    base_value: int,
    xs: Sequence[int],
    out_path: Optional[Path] = None,
    atomic_path: Optional[Path] = None,
) -> Dict[str, Any]:
    report = run_series_field_inspect(op, base_value, xs, atomic_path=atomic_path)
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "data" / "field_inspect_report.json"
    stored = store_report(report, Path(out_path))
    d = report_to_dict(report)
    d["stored_path"] = stored
    return d
