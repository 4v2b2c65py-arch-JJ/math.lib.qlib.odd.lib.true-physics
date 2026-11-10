"""Relativity resonance set as base plate (Einstein playbook).

Uses existing physics constants (scipy CODATA) as reference:
- c from scipy.constants
- Apply SR corrections: Lorentz factor gamma, time dilation, length contraction,
  relativistic mass/energy (E=γmc^2), velocity addition.
- Compare conventional/NN outputs against relativistic-corrected values and
  apply corrections, storing results by hash for the math tree/event loop.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import scipy
    from scipy import constants as C
    C_CONST = float(C.c)
except Exception:
    C_CONST = 299792458.0

from . import (
    GridRow,
    HashBaseStore,
    MathOp,
    MathTree,
    PlankOutput,
    TimesPlankGrid,
    build_grid_ops,
    build_row_op,
    process_math_event_loop,
    tree_key_hash,
)


@dataclass
class RelativisticCorrection:
    beta: float
    gamma: float
    time_dilation_factor: float  # dt = gamma * dt0
    length_contraction_factor: float  # L = L0 / gamma
    relativistic_energy_factor: float  # E = gamma * m * c^2 (relative factor)
    correction_applied: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class CorrectionReport:
    base_c: float
    corrections: List[Dict[str, Any]]
    summary: Dict[str, Any]
    generated_at_ns: int = field(default_factory=time.time_ns)


def lorentz_gamma(v: float, c: float = C_CONST) -> float:
    if c <= 0:
        return float("inf")
    beta = abs(float(v)) / float(c)
    if beta >= 1.0:
        beta = 1.0 - 1e-18  # avoid singularity; treat as ultra-relativistic
    beta2 = beta * beta
    return 1.0 / math.sqrt(max(1e-18, 1.0 - beta2))


def apply_sr_corrections(v: float, c: float = C_CONST) -> RelativisticCorrection:
    gamma = lorentz_gamma(v, c)
    beta = abs(float(v)) / float(c)
    notes: List[str] = []
    notes.append("Einstein SR base plate: c constant, Lorentz transformations")
    if beta > 0.1:
        notes.append(f"non-negligible relativistic regime beta={beta:.6f}")
    return RelativisticCorrection(
        beta=beta,
        gamma=gamma,
        time_dilation_factor=gamma,
        length_contraction_factor=1.0 / gamma if gamma > 0 else float("inf"),
        relativistic_energy_factor=gamma,
        correction_applied=True,
        notes=notes,
    )


def relativistic_velocity_add(u: float, v: float, c: float = C_CONST) -> float:
    """Einstein velocity addition (1D)."""
    if c <= 0:
        return u + v
    uv = float(u) + float(v)
    denom = 1.0 + (float(u) * float(v)) / (c * c)
    if abs(denom) < 1e-18:
        return math.copysign(c, uv) if uv != 0 else 0.0
    w = uv / denom
    # clamp to |w| < c
    if w >= c:
        w = c * (1.0 - 1e-18)
    if w <= -c:
        w = -c * (1.0 - 1e-18)
    return w


def _hash_correction(key: str, val: float, gamma: float) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(key.encode())
    h.update(b"|")
    h.update(str(val).encode())
    h.update(b"|")
    h.update(str(gamma).encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def correct_plank_output(out: PlankOutput, corr: RelativisticCorrection) -> PlankOutput:
    """Apply relativity correction to a times-plank output value (hash-addressed)."""
    g = corr.gamma
    # Apply correction relative to base: scale value by gamma with sign preserved
    corrected_val = int(round(float(out.value) * g))
    new_hash = _hash_correction(f"rel:{out.agent_id}:{out.address.x},{out.address.y},{out.address.z}", corrected_val, g)
    return PlankOutput(
        value=corrected_val,
        address=out.address,
        agent_id=out.agent_id,
        hash=new_hash,
    )


def correct_grid_row(row: GridRow, corr: RelativisticCorrection) -> GridRow:
    new_cells = [correct_plank_output(c, corr) for c in row.cells]
    # recompute base hash for corrected row
    h = hashlib.blake2b(digest_size=8)
    h.update(b"rel_row|")
    h.update(str(row.base_hash).encode())
    h.update(b"|")
    h.update(str(corr.gamma).encode())
    return GridRow(base_hash=int.from_bytes(h.digest(), "little", signed=False), cells=new_cells)


def correct_times_plank_grid(grid: TimesPlankGrid, velocity: float, c: float = C_CONST) -> Tuple[TimesPlankGrid, RelativisticCorrection]:
    """Apply Einstein-relativity corrections to entire grid outputs."""
    corr = apply_sr_corrections(velocity, c)
    new_grid = TimesPlankGrid()
    new_grid.log = list(grid.log)
    new_grid.log.append(f"rel_correction applied: v={velocity} beta={corr.beta:.9f} gamma={corr.gamma:.12f} base_c={c}")
    new_grid.num_positive = 0
    new_grid.num_negative = 0
    for row in grid.rows:
        crow = correct_grid_row(row, corr)
        for cell in crow.cells:
            if cell.value >= 0:
                new_grid.num_positive += 1
            else:
                new_grid.num_negative += 1
        new_grid.rows.append(crow)
    return new_grid, corr


def correct_via_event_loop(
    a: int,
    b: int,
    op: MathOp = MathOp.MUL,
    velocity: float = 0.0,
    c: float = C_CONST,
) -> Dict[str, Any]:
    """Use hash-addressed event loop: build tree node, apply relativity correction, store corrected node."""
    tree = MathTree()
    from . import CalcEvent, process_math_event_loop
    process_math_event_loop(tree, [CalcEvent(op, a, b)])
    key = tree_key_hash(op, a, b)
    node = tree.nodes.get(key)
    if node is None:
        return {"error": "node not found"}
    corr = apply_sr_corrections(velocity, c)
    crow = correct_grid_row(node.row, corr)
    node_corrected = type(node)(key_hash=key, op=op, a=a, b=b, row=crow, version=node.version + 1)
    tree.nodes[key + 0x13579BDF] = node_corrected  # store corrected under derived hash address
    tree.event_log.append(f"rel_correct key={key} -> corrected_key={key+0x13579BDF} gamma={corr.gamma:.12f}")
    return {
        "original_key": key,
        "corrected_key": key + 0x13579BDF,
        "correction": asdict(corr),
        "original_row_hash": node.row.base_hash,
        "corrected_row_hash": crow.base_hash,
        "original_values": [c.value for c in node.row.cells],
        "corrected_values": [c.value for c in crow.cells],
        "event_log": tree.event_log,
    }


def build_relativity_corrected_report(
    base_value: int = 8,
    xs: Sequence[int] = (1, 10, 100, 1000, 8000),
    op: MathOp = MathOp.MUL,
    velocities: Sequence[float] = (0.0, 0.1 * C_CONST, 0.5 * C_CONST, 0.8 * C_CONST),
    c: float = C_CONST,
) -> CorrectionReport:
    """Build correction report comparing Einstein playbook corrections across velocities."""
    calcs = [(op, base_value, int(x)) for x in xs]
    grid = build_grid_ops(calcs)
    corrections = []
    for v in velocities:
        cgrid, corr = correct_times_plank_grid(grid, float(v), c)
        corrections.append(
            {
                "velocity": float(v),
                "beta": corr.beta,
                "gamma": corr.gamma,
                "time_dilation_factor": corr.time_dilation_factor,
                "length_contraction_factor": corr.length_contraction_factor,
                "corrected_rows": len(cgrid.rows),
                "corrected_outputs": sum(len(r.cells) for r in cgrid.rows),
                "sample_corrected_values": [r.cells[0].value if r.cells else None for r in cgrid.rows],
            }
        )
    return CorrectionReport(
        base_c=c,
        corrections=corrections,
        summary={
            "scipy_version": getattr(scipy, "__version__", "unknown"),
            "base_value": base_value,
            "series": list(xs),
            "op": op.value,
            "velocities_tested": len(velocities),
        },
    )
