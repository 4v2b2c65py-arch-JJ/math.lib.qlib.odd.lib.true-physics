"""ASCII video generation driven by mathlib "smart calculator" (times-plank/tree/matrix),
not traditional fixed gridding.

CSS Rule C (visual/calc output constraint):
- Tick precision: every visual scalar must align to a positive or negative tick step
  (non-zero signed quantum) so values are discretely precise.
- Memory equality: allocated memory for result output must equal memory of input for the
  visual buffer (len(output_frames) == len(input_frames) and per-frame/cell buffer sizes
  match), so the result is exactly as envisioned.
These rules are enforced for generated visual outputs.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from . import (
    MathOp,
    SpeculativeMatrix,
    TimesPlankGrid,
    await_outs,
    build_grid_ops,
    cell_feature_vector,
    run_series_op,
    speculative_matrix_from_grid,
    times_plank_divide,
    times_plank_multiply,
)


# use pi as css div (division base for CSS Rule C visual/calc grid)
PI_CSS_DIV = math.pi


@dataclass
class CSSRuleCConfig:
    tick: float = 1.0  # magnitude of tick step; sign gives +/- tick
    css_div: float = PI_CSS_DIV  # pi as css div (division base)
    enforce_memory_equality: bool = True

    def normalized_tick(self) -> float:
        t = float(self.tick)
        if t == 0.0:
            t = 1.0
        return t

    def normalized_div(self) -> float:
        d = float(self.css_div)
        if d == 0.0:
            d = PI_CSS_DIV
        return d


def solve_trajectory_tick(
    base_tick: float,
    frame_index: int,
    refresh_ns: Optional[Sequence[int]] = None,
    total_frames: int = 1,
    css_div: float = PI_CSS_DIV,
) -> float:
    """Solve trajectory tick per frame to account for variable refresh.
    Uses per-frame refresh durations (ns) when provided; otherwise derives a
    deterministic progression over the trajectory so tick adapts across frames.
    Uses pi as css div for effective tick quantization.
    """
    t = abs(float(base_tick)) if base_tick not in (0, 0.0) else 1.0
    d = abs(float(css_div)) if css_div not in (0, 0.0) else PI_CSS_DIV
    n = max(1, int(total_frames))

    if refresh_ns and len(refresh_ns) > 0:
        # variable refresh: scale tick by frame refresh relative to mean refresh
        vals = [max(1, int(r)) for r in refresh_ns]
        mean_r = (sum(vals) / len(vals)) or 1.0
        idx = min(frame_index, len(vals) - 1)
        factor = vals[idx] / mean_r
        solved = t * factor
        q = quantize_tick(solved, t, css_div=d)
        return q if q != 0.0 else (t / d if solved >= 0 else -t / d)

    # deterministic trajectory when explicit refresh timing not supplied
    progress = frame_index / max(1, n - 1) if n > 1 else 0.0
    factor = 0.5 + progress
    solved = t * factor
    q = quantize_tick(solved, t, css_div=d)
    return q if q != 0.0 else (t / d)


def quantize_tick(value: float, tick: float, css_div: float = PI_CSS_DIV) -> float:
    """Align value to nearest signed tick multiple (positive or negative tick),
    using pi as css div (division base) for the effective step."""
    d = abs(float(css_div)) if css_div not in (0, 0.0) else PI_CSS_DIV
    t = abs(float(tick)) if tick not in (0, 0.0) else 1.0
    step = t / d  # pi as css div
    if step == 0.0:
        step = 1.0 / PI_CSS_DIV
    if value == 0.0:
        return 0.0
    q = round(value / step) * step
    if q == 0.0:
        q = step if value > 0 else -step
    return float(q)


def apply_tick_to_matrix(matrix: SpeculativeMatrix, tick: float, css_div: float = PI_CSS_DIV) -> SpeculativeMatrix:
    t = abs(tick) if tick not in (0, 0.0) else 1.0
    d = abs(css_div) if css_div not in (0, 0.0) else PI_CSS_DIV
    new_cells = []
    for c in matrix.cells:
        v = quantize_tick(c.value, t, css_div=d)
        new_cells.append(type(c)(row=c.row, col=c.col, value=v, hash=c.hash, category=c.category, score=c.score))
    return SpeculativeMatrix(rows=matrix.rows, cols=matrix.cols, cells=new_cells)


@dataclass
class MemoryAllocationReport:
    input_units: int
    output_units: int
    equal: bool
    detail: str


def enforce_memory_equality(input_buffers: Sequence[Sequence], output_buffers: List[list], config: CSSRuleCConfig) -> MemoryAllocationReport:
    """Allocate output memory equal to input memory; pad/trim output buffers to match input."""
    in_units = len(input_buffers)
    out_units = len(output_buffers)
    # match frame count
    if config.enforce_memory_equality:
        if out_units < in_units:
            # pad with empty frames allocated equal to corresponding input frame size
            for i in range(out_units, in_units):
                output_buffers.append([None] * len(input_buffers[i]))
        elif out_units > in_units:
            del output_buffers[in_units:]
        # match per-frame cell allocation
        for i, inp in enumerate(input_buffers):
            need = len(inp)
            if len(output_buffers[i]) < need:
                output_buffers[i].extend([None] * (need - len(output_buffers[i])))
            elif len(output_buffers[i]) > need:
                del output_buffers[i][need:]
    final_out = len(output_buffers)
    equal = final_out == in_units and all(len(output_buffers[i]) == len(input_buffers[i]) for i in range(min(in_units, final_out)))
    return MemoryAllocationReport(
        input_units=sum(len(b) for b in input_buffers),
        output_units=sum(len(b) for b in output_buffers),
        equal=equal,
        detail=f"frames_in={in_units} frames_out={final_out} cells_equal={equal}",
    )


def _canvas(width: int, height: int, fill: str = " ") -> List[List[str]]:
    return [[fill for _ in range(width)] for _ in range(height)]


def _canvas_to_str(c: List[List[str]]) -> str:
    return "\n".join("".join(row) for row in c)


def _intensity_char(v: float, vmax: float) -> str:
    ramp = " .:-=+*#%@"
    if vmax <= 0:
        return " "
    idx = int((abs(v) / vmax) * (len(ramp) - 1))
    idx = max(0, min(len(ramp) - 1, idx))
    ch = ramp[idx]
    return ch if v >= 0 else ch.lower() if ch.isalpha() else ("-" if v < 0 else ch)


def render_matrix_frame(matrix: SpeculativeMatrix, width: int, height: int, tick: float, frame_index: int = 0, css_div: float = PI_CSS_DIV) -> str:
    """Render one ASCII frame from speculative matrix using tick-precise mapped values.
    Uses pi as css div for quantization/division base.
    """
    m = apply_tick_to_matrix(matrix, tick, css_div=css_div)
    c = _canvas(width, height, " ")
    if not m.cells:
        return _canvas_to_str(c)
    vmax = max((abs(cell.value) for cell in m.cells), default=1.0) or 1.0
    # map matrix (row,col) into canvas; smart mapping: use value/hash-driven placement
    for cell in m.cells:
        # normalized positions influenced by speculative value/hash (not plain uniform grid only)
        nx = (cell.col + (cell.hash % 97) / 97.0) / max(1, m.cols)
        ny = (cell.row + ((cell.hash // 97) % 97) / 97.0) / max(1, m.rows)
        # animate slightly by frame_index using hash (no sin/cos; hash-based resonance)
        hx = ((cell.hash ^ (frame_index * 0x9E3779B1)) & 0xFFFF) / 0xFFFF
        hy = (((cell.hash >> 16) ^ (frame_index * 0x85EBCA77)) & 0xFFFF) / 0xFFFF
        dx = hx * 2.0 - 1.0
        dy = hy * 2.0 - 1.0
        x = int((nx + 0.03 * dx) * (width - 1))
        y = int((ny + 0.03 * dy) * (height - 1))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        c[y][x] = _intensity_char(cell.value, vmax)
    # overlay tick signature (show pi css div)
    sig = f"t{('+' if tick>=0 else '')}{tick:g}/pi f{frame_index}"
    for i, ch in enumerate(sig[:width]):
        c[0][i] = ch
    return _canvas_to_str(c)


def generate_ascii_video_from_series(
    op: MathOp,
    base_value: int,
    xs: Sequence[int],
    *,
    width: Optional[int] = None,
    height: int = 24,
    tick: float = 1.0,
    category: str = "calc",
    config: Optional[CSSRuleCConfig] = None,
    refresh_ns: Optional[Sequence[int]] = None,
) -> dict:
    """Generate ASCII 'video' (sequence of frames) from a smart-calculator series.

    Uses run_series_op -> speculative_matrix_from_grid per step window (not traditional gridding).
    Enforces CSS Rule C: tick quantization and memory allocation equality.
    Solves trajectory ticks per frame to account for variable refresh.
    """
    cfg = config or CSSRuleCConfig(tick=tick)
    tick_n = cfg.normalized_tick()
    css_div_n = cfg.normalized_div()
    term_w = shutil.get_terminal_size((80, 24)).columns
    w = int(width or min(80, max(20, term_w)))
    h = int(max(6, height))

    # Input buffer represents envisioned/calculation inputs per frame
    input_buffers: List[List[int]] = [[int(base_value), int(x)] for x in xs]
    # Pre-allocate output buffers equal to input memory (one frame buffer per input)
    output_frames: List[List[Optional[str]]] = [[None] for _ in input_buffers]

    frames: List[str] = []
    trajectory_ticks: List[float] = []
    observed_refresh_ns: List[int] = list(refresh_ns) if refresh_ns is not None else []
    # progressive smart-calculator windows: each frame uses accumulated series calculations
    for i in range(len(xs)):
        window = list(xs[: i + 1])
        run = run_series_op(op, base_value, window)
        # capture per-window timing as observed refresh input when not explicitly provided
        if refresh_ns is None:
            observed_refresh_ns.append(int(getattr(run.timing, "total_ns", 0) or 0))
        ref_seq = refresh_ns if refresh_ns is not None else observed_refresh_ns
        frame_tick = solve_trajectory_tick(tick_n, i, refresh_ns=ref_seq, total_frames=len(xs), css_div=css_div_n)
        trajectory_ticks.append(frame_tick)

        matrix = speculative_matrix_from_grid(run.grid, category=category)
        frame = render_matrix_frame(matrix, w, h, frame_tick, frame_index=i, css_div=css_div_n)
        output_frames[i][0] = frame
        frames.append(frame)

    mem = enforce_memory_equality(input_buffers, output_frames, cfg)
    # ensure returned frames correspond to equalized output buffers
    eq_frames = [("" if (bf[0] is None) else str(bf[0])) for bf in output_frames]

    sym = "x" if op == MathOp.MUL else "/"
    tick_summary = f"{trajectory_ticks[0]:g}..{trajectory_ticks[-1]:g}" if trajectory_ticks else f"{tick_n:g}"
    header = f"ASCII-VID {base_value}{sym}{xs[0] if xs else ''}..{xs[-1] if xs else ''} frames={len(eq_frames)} tick={tick_summary} css_div=pi mem_equal={mem.equal}"
    return {
        "header": header,
        "frames": eq_frames,
        "frame_count": len(eq_frames),
        "width": w,
        "height": h,
        "tick": tick_n,
        "css_div": css_div_n,
        "css_div_name": "pi",
        "trajectory_ticks": trajectory_ticks,
        "observed_refresh_ns": observed_refresh_ns,
        "memory": {
            "input_units": mem.input_units,
            "output_units": mem.output_units,
            "equal": mem.equal,
            "detail": mem.detail,
        },
        "css_rule_c": {
            "tick_precision": True,
            "memory_allocation_equal_to_input": mem.equal,
            "pi_as_css_div": True,
        },
        "op": op.value,
        "base": base_value,
        "xs": list(xs),
    }


def play_ascii_video(result: dict, fps: float = 4.0) -> str:
    """Return a single text payload representing the video (frame-separated) for display/logging."""
    sep = "\n" + ("-" * max(10, int(result.get("width", 40)))) + "\n"
    return result.get("header", "ASCII-VID") + "\n" + sep.join(result.get("frames", []))
