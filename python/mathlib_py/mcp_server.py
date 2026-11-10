"""MCP server exposing mathlib as a smart calculator and ASCII video generator.

Run: python -m mathlib_py.mcp_server
Exposes tools for times-plank mul/div, series, speculative matrix/category sort,
hash-addressed math tree event loop, and CSS-Rule-C ASCII video generation.
"""
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from . import (
    HashBaseStore,
    MathOp,
    MathWeightModel,
    array_field,
    build_grid_ops,
    division_table,
    integrate_category_search,
    process_math_event_loop,
    run_series_op,
    sort_by_category,
    speculative_matrix_from_grid,
    times_plank_divide,
    times_plank_multiply,
    times_table,
)
from .ascii_vid import CSSRuleCConfig, generate_ascii_video_from_series, play_ascii_video, quantize_tick

mcp = FastMCP("mathlib-smart-calculator")


def _op(op: str) -> MathOp:
    return MathOp.DIV if str(op).lower() in ("div", "/", "division") else MathOp.MUL


@mcp.tool()
def smart_calculate(a: int, b: int, op: str = "mul", tick: float = 1.0) -> dict:
    """Times-plank smart calculation (mul/div) with agent-sorted addressed outputs and tick-precise values."""
    mo = _op(op)
    outs = times_plank_divide(a, b) if mo == MathOp.DIV else times_plank_multiply(a, b)
    return {
        "op": mo.value,
        "a": a,
        "b": b,
        "tick": abs(tick) if tick else 1.0,
        "outputs": [
            {
                "value": o.value,
                "tick_value": quantize_tick(float(o.value), abs(tick) if tick else 1.0),
                "address": {"x": o.address.x, "y": o.address.y, "z": o.address.z},
                "agent_id": o.agent_id,
                "hash": o.hash,
            }
            for o in outs
        ],
    }


@mcp.tool()
def generate_ascii_video(
    op: str = "mul",
    base_value: int = 8,
    xs: Optional[list] = None,
    width: int = 72,
    height: int = 20,
    tick: float = 1.0,
    refresh_ns: Optional[list] = None,
    fps: float = 4.0,
    category: str = "timeline",
    as_text: bool = True,
) -> dict:
    """Generate ASCII video via smart calculator (not traditional gridding).
    Enforces CSS Rule C: tick-aligned visual output and memory allocation equal to input.
    Solves trajectory ticks per frame to account for variable refresh (from refresh_ns or observed timings).
    Returns frames and (optionally) concatenated text payload.
    """
    if xs is None:
        xs = list(range(1, 13))
    mo = _op(op)
    cfg = CSSRuleCConfig(tick=tick, enforce_memory_equality=True)
    result = generate_ascii_video_from_series(
        mo,
        int(base_value),
        [int(x) for x in xs],
        width=width,
        height=height,
        tick=tick,
        category=category,
        config=cfg,
        refresh_ns=[int(r) for r in refresh_ns] if refresh_ns is not None else None,
    )
    result["fps"] = fps
    if as_text:
        result["text"] = play_ascii_video(result, fps=fps)
    return result


@mcp.tool()
def times_table_tool(base_value: int = 8, n: int = 12) -> dict:
    """Generate times table grid (smart calculator)."""
    g = times_table(base_value, n)
    return {"rows": len(g.rows), "array_field": array_field(g), "log": g.log[:20]}


@mcp.tool()
def division_table_tool(dividend: int = 100, n: int = 10) -> dict:
    """Generate division table grid (smart calculator)."""
    g = division_table(dividend, n)
    return {"rows": len(g.rows), "array_field": array_field(g), "log": g.log[:20]}


@mcp.tool()
def math_tree_event_loop(ops: list) -> dict:
    """Run hash-addressed event loop over calculations; compares/overwrites tree nodes by hash.
    ops: [{"op":"mul"|"div","a":int,"b":int}, ...]
    """
    from . import CalcEvent, MathTree
    tree = MathTree()
    events = [CalcEvent(_op(e.get("op", "mul")), int(e["a"]), int(e["b"])) for e in ops]
    process_math_event_loop(tree, events)
    return {
        "nodes": len(tree.nodes),
        "event_log": tree.event_log,
        "versions": {str(k): v.version for k, v in tree.nodes.items()},
    }


@mcp.tool()
def speculative_category_refine(base_value: int = 8, n: int = 12, op: str = "mul", category: str = "timeline") -> dict:
    """Build speculative matrix from smart-calculator grid, category-sort (image-search-like), and refine model weights/nodes."""
    mo = _op(op)
    g = build_grid_ops([(mo, base_value, i) for i in range(1, max(1, n) + 1)])
    m = speculative_matrix_from_grid(g, category=category)
    ranked = sort_by_category(m, category)
    model = MathWeightModel(0.1)
    res = integrate_category_search(model, m, category, adapter=None)
    return {"matrix_cells": len(m.cells), "ranked": len(ranked), **res}


@mcp.tool()
def custom_runtime(
    cmd: str = "status",
    config: str = "",
    source: str = "",
    target: str = "",
    device: str = "",
    out: str = "",
    data: Optional[list] = None,
    file: str = "",
    deep: bool = False,
    enable_mocks: bool = False,
    force_load: bool = False,
    mock_devices: Optional[list] = None,
    force_load_functions: Optional[list] = None,
    require_null_place_turn: bool = True,
    turn: Optional[dict] = None,
    slot: Optional[str] = None,
) -> dict:
    """Custom runtime: cmd=status|probe|detect|transfer|force-load|place-turn. Config as JSON or @path. Supports device mocks, force-load, and place_turn requiring null slot."""
    from .custom_runtime import CustomRuntime
    rt = CustomRuntime.from_config(config or None)
    if source: rt.config.default_source = source
    if target: rt.config.default_target = target
    if device: rt.config.device = device
    if enable_mocks:
        rt.config.enable_mocks = True
    if mock_devices is not None:
        rt.config.mock_devices = mock_devices
    if force_load:
        rt.config.force_load = True
    if force_load_functions is not None:
        rt.config.force_load_functions = force_load_functions
    rt.config.require_null_for_place_turn = bool(require_null_place_turn)
    if cmd == "probe":
        return rt.probe(deep=deep)
    if cmd == "detect":
        return rt.detect_cards(deep=deep)
    if cmd == "force-load":
        return rt.force_load()
    if cmd == "place-turn":
        slot_val = None if slot in (None, "", "null", "None") else slot
        return rt.place_turn(turn=turn, slot=slot_val)
    if cmd == "transfer":
        payload = data
        if file:
            payload = {"file": file}
        return rt.transfer(data=payload, source=source or None, target=target or None, out=out or None, device=device or None, deep=deep)
    return rt.status()


@mcp.tool()
def gpu_backends(prefer: str = "") -> dict:
    """Probe/connect Vulkan and CUDA GPU backends and return unified status/bridge info."""
    from .gpu_backends import connect_backends
    pref = prefer if prefer in ("vulkan", "cuda") else None
    return connect_backends(prefer=pref)


@mcp.tool()
def gpu_vram_transfer(data: Optional[list] = None, file: str = "", source: str = "auto", target: str = "host", out: str = "", device: str = "") -> dict:
    """Host-staged VRAM/buffer assistance transfer. Supports host payloads and CUDA staging via Leviathan env; direct cross-vendor RX580->CUDA VRAM offload is not possible."""
    from .gpu_backends import transfer_vram
    payload = data
    if file:
        payload = {"file": file}
    return transfer_vram(data=payload, source=source, target=target, out=out or None, device=device or None, deep=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
