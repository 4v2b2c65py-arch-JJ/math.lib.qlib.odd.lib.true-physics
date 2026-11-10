"""Python wrapper for timeline mathlib (times-plank, agent sorting, timeline clock).

Refactored calculations use signature base(a, b, store=...) so each calculation
can persist hash data per run into a base store. Supports mul/div and hash-addressed
event loop to compare/overwrite a math tree structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class MathOp(str, Enum):
    MUL = "mul"
    DIV = "div"


@dataclass(frozen=True)
class Address:
    x: int
    y: int
    z: int


@dataclass
class QbitStructure:
    """Lightweight qbit-like state annotation for an output (for array handling/foreseen entries)."""
    alpha: float  # amplitude/probability weight for + branch
    beta: float   # amplitude/probability weight for - branch
    phase: float  # derived phase from hash
    label: str = "|psi>"


@dataclass
class PlankOutput:
    value: int
    address: Address
    agent_id: int
    hash: int
    category: str = "calc"
    qbit: Optional[QbitStructure] = None
    conjecture: bool = False
    stored_hash_value: Optional[int] = None  # when conjecture rule stores hash instead of conventional value


@dataclass
class GridRow:
    base_hash: int
    cells: List[PlankOutput] = field(default_factory=list)


@dataclass
class TimesPlankGrid:
    rows: List[GridRow] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    num_positive: int = 0
    num_negative: int = 0


@dataclass
class CalcTiming:
    per_row_ns: List[int] = field(default_factory=list)
    total_ns: int = 0


@dataclass
class TimedRun:
    grid: TimesPlankGrid
    timing: CalcTiming


class HashBaseStore:
    """Base(num) store: holds hash/data per calculation for hash decoding."""

    def __init__(self, base_num: int = 10):
        self.base_num = base_num
        self.records: List[dict] = []

    def record(self, a: int, b: int, row_hash: int, outputs: Sequence[PlankOutput], elapsed_ns: int, op: MathOp = MathOp.MUL) -> None:
        self.records.append(
            {
                "op": op.value,
                "a": a,
                "b": b,
                "base_hash": row_hash,
                "elapsed_ns": elapsed_ns,
                "outputs": [
                    {
                        "value": o.value,
                        "agent_id": o.agent_id,
                        "hash": o.hash,
                        "address": (o.address.x, o.address.y, o.address.z),
                    }
                    for o in outputs
                ],
            }
        )

    def summary(self) -> dict:
        return {"base_num": self.base_num, "rows": len(self.records), "outputs": sum(len(r["outputs"]) for r in self.records)}


def standard_multiply(a: int, b: int) -> int:
    return a * b


def standard_divide(a: int, b: int) -> int:
    return 0 if b == 0 else int(a / b)  # trunc toward zero like Rust i64


def digit_addresses(n: int) -> List[Address]:
    s = str(abs(n))
    return [Address(i, int(c), 0) for i, c in enumerate(s)]


def score_hash_op(op: MathOp, a: int, b: int, agent_id: int, value: int, address: Address) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{op.value}|{a}|{b}|{agent_id}|{value}|{address.x},{address.y},{address.z}".encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def score_hash(a: int, b: int, agent_id: int, value: int, address: Address) -> int:
    return score_hash_op(MathOp.MUL, a, b, agent_id, value, address)


def _primary_digit_transform(a: int, b: int) -> Optional[int]:
    digits = digit_addresses(a)
    if not digits:
        return None
    s = "".join(str(d.y * b) for d in digits)
    try:
        v = int(s)
    except ValueError:
        return None
    return -v if a < 0 else v


def _primary_digit_div_transform(a: int, b: int) -> Optional[int]:
    if b == 0:
        return None
    digits = digit_addresses(a)
    if not digits:
        return None
    s = "".join(str(abs(int(d.y / b))) for d in digits) or "0"
    try:
        v = int(s)
    except ValueError:
        return None
    sign_neg = (a < 0) ^ (b < 0)
    return -v if sign_neg else v


def times_plank_multiply(a: int, b: int) -> List[PlankOutput]:
    outputs: List[PlankOutput] = []
    std = standard_multiply(a, b)
    op = MathOp.MUL

    v0 = _primary_digit_transform(a, b)
    if v0 is not None:
        addr = Address(0, 0, 0)
        outputs.append(PlankOutput(v0, addr, 0, score_hash_op(op, a, b, 0, v0, addr)))

    addr_std = Address(1, 0, 0)
    outputs.append(PlankOutput(std, addr_std, 1, score_hash_op(op, a, b, 1, std, addr_std)))

    digits = digit_addresses(a)
    if len(digits) >= 2:
        products = [d.y * b for d in digits]
        sum_rest = sum(products[1:])
        try:
            v_a = int(f"{abs(products[0])}{abs(sum_rest)}")
            if a < 0:
                v_a = -v_a
            addr = Address(2, 0, 1)
            outputs.append(PlankOutput(v_a, addr, 2, score_hash_op(op, a, b, 2, v_a, addr)))
        except ValueError:
            pass
        head_sum = products[0] + (products[1] if len(products) > 1 else 0)
        tail = products[-1]
        try:
            v_b = int(f"{abs(head_sum)}{abs(tail) % 10}")
            if a < 0:
                v_b = -v_b
            addr = Address(3, 0, 2)
            outputs.append(PlankOutput(v_b, addr, 3, score_hash_op(op, a, b, 3, v_b, addr)))
        except ValueError:
            pass
    return CONJECTURE_RULE.apply(outputs, op=op)


def times_plank_divide(a: int, b: int) -> List[PlankOutput]:
    outputs: List[PlankOutput] = []
    std = standard_divide(a, b)
    op = MathOp.DIV

    v0 = _primary_digit_div_transform(a, b)
    if v0 is not None:
        addr = Address(0, 0, 10)
        outputs.append(PlankOutput(v0, addr, 0, score_hash_op(op, a, b, 0, v0, addr)))

    addr_std = Address(1, 0, 10)
    outputs.append(PlankOutput(std, addr_std, 1, score_hash_op(op, a, b, 1, std, addr_std)))

    if b != 0:
        digits = digit_addresses(a)
        sign_neg = (a < 0) ^ (b < 0)
        if len(digits) >= 2:
            quots = [int(d.y / b) for d in digits]
            rems = [d.y % b for d in digits]
            sum_rem = sum(rems)
            try:
                v_a = int("".join(str(abs(q)) for q in quots) + str(abs(sum_rem)))
                if sign_neg:
                    v_a = -v_a
                addr = Address(2, 0, 11)
                outputs.append(PlankOutput(v_a, addr, 2, score_hash_op(op, a, b, 2, v_a, addr)))
            except ValueError:
                pass
            rem_total = a % b
            try:
                v_b = int(f"{abs(std)}{abs(rem_total)}")
                if sign_neg:
                    v_b = -v_b
                addr = Address(3, 0, 12)
                outputs.append(PlankOutput(v_b, addr, 3, score_hash_op(op, a, b, 3, v_b, addr)))
            except ValueError:
                pass
        else:
            rem_total = a % b
            try:
                v_b = int(f"{abs(std)}{abs(rem_total)}")
                if sign_neg:
                    v_b = -v_b
                addr = Address(3, 0, 12)
                outputs.append(PlankOutput(v_b, addr, 3, score_hash_op(op, a, b, 3, v_b, addr)))
            except ValueError:
                pass
    return CONJECTURE_RULE.apply(outputs, op=op)


def qbit_from_hash(h: int, value: int) -> QbitStructure:
    # deterministic pseudo-qbit amplitudes derived from hash/value for structured array sorting/forecasting
    u = (h & 0xFFFFFFFF) / 0xFFFFFFFF
    alpha = math.sqrt(max(0.0, min(1.0, u)))
    beta = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    phase = ((h >> 32) % 360) * math.pi / 180.0
    # flip emphasis for negative values
    if value < 0:
        alpha, beta = beta, alpha
    return QbitStructure(alpha=alpha, beta=beta, phase=phase, label="|psi>")


def _categorize_output(op: MathOp, agent_id: int, address: Address, conjecture: bool) -> str:
    kind = "conjecture" if conjecture else "result"
    return f"{op.value}:{kind}:agent{agent_id}:z{address.z}"


def enrich_output(o: PlankOutput, op: MathOp = MathOp.MUL, conjecture: bool = False) -> PlankOutput:
    o.category = _categorize_output(op, o.agent_id, o.address, conjecture)
    o.qbit = qbit_from_hash(o.hash, o.value)
    o.conjecture = conjecture
    if conjecture:
        # Rule: conjectures store val as hash instead of conventional arithmetic value
        o.stored_hash_value = o.hash
    return o


def sort_agents_by_score(outputs: List[PlankOutput]) -> None:
    outputs.sort(key=lambda o: (o.hash, o.agent_id))


def sort_outputs_by_category(outputs: List[PlankOutput]) -> List[PlankOutput]:
    """Sort array by category then hash to keep good arrays and handle multiple/foreseen entries."""
    return sorted(outputs, key=lambda o: (o.category, o.hash, o.agent_id, o.address.z, o.address.x))


def group_outputs_by_category(outputs: List[PlankOutput]) -> Dict[str, List[PlankOutput]]:
    grouped: Dict[str, List[PlankOutput]] = {}
    for o in sort_outputs_by_category(outputs):
        grouped.setdefault(o.category, []).append(o)
    return grouped


@dataclass
class ConjectureRule:
    """Rule: treat conjectures as hash-addressed hypotheses; store val as hash (not conventional arithmetic)."""
    enabled: bool = True
    note: str = "Conjecture values are stored as hash identifiers; conventional value is interpretive."

    def apply(self, outputs: List[PlankOutput], op: MathOp = MathOp.MUL) -> List[PlankOutput]:
        treated: List[PlankOutput] = []
        for o in outputs:
            is_conj = (o.agent_id != 1)  # agent 1 is standard/reference; others treated as conjectural/spatial hypotheses
            treated.append(enrich_output(o, op=op, conjecture=is_conj if self.enabled else False))
        return treated


CONJECTURE_RULE = ConjectureRule(enabled=True)


def _row_hash(op: MathOp, a: int, b: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{op.value}|{a}|{b}".encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def build_row_op(op: MathOp, a: int, b: int) -> GridRow:
    outputs = times_plank_multiply(a, b) if op == MathOp.MUL else times_plank_divide(a, b)
    sort_agents_by_score(outputs)
    return GridRow(_row_hash(op, a, b), outputs)


def build_row(a: int, b: int) -> GridRow:
    return build_row_op(MathOp.MUL, a, b)


def build_grid_ops(calcs: Iterable[Tuple[MathOp, int, int]]) -> TimesPlankGrid:
    grid = TimesPlankGrid()
    for op, a, b in calcs:
        row = build_row_op(op, a, b)
        # keep per-row outputs as good arrays: sorted by category with qbit/conjecture metadata
        row.cells = sort_outputs_by_category(row.cells)
        for cell in row.cells:
            if cell.value >= 0:
                grid.num_positive += 1
            else:
                grid.num_negative += 1
        sym = "x" if op == MathOp.MUL else "/"
        grid.log.append(f"row base_hash={row.base_hash} outs={len(row.cells)} for {a}{sym}{b}")
        grid.rows.append(row)
    return grid


def build_grid(pairs: Iterable[Tuple[int, int]]) -> TimesPlankGrid:
    return build_grid_ops((MathOp.MUL, a, b) for a, b in pairs)


def times_table(base_value: int, n: int) -> TimesPlankGrid:
    return build_grid((base_value, i) for i in range(1, max(0, n) + 1))


def division_table(dividend: int, n: int) -> TimesPlankGrid:
    return build_grid_ops((MathOp.DIV, dividend, i) for i in range(1, max(0, n) + 1))


def base(a: int, b: int, store: Optional[HashBaseStore] = None, op: MathOp = MathOp.MUL) -> GridRow:
    """Signature calculation: base(num, op)/store. Uses times-plank and records hash per calc."""
    t0 = time.perf_counter_ns()
    row = build_row_op(op, a, b)
    elapsed = time.perf_counter_ns() - t0
    if store is not None:
        store.record(a, b, row.base_hash, row.cells, elapsed, op=op)
    return row


def run_series_op(op: MathOp, base_value: int, xs: Sequence[int], store: Optional[HashBaseStore] = None) -> TimedRun:
    grid = TimesPlankGrid()
    timing = CalcTiming()
    t0 = time.perf_counter_ns()
    sym = "x" if op == MathOp.MUL else "/"
    for x in xs:
        r0 = time.perf_counter_ns()
        row = base(base_value, int(x), store=store, op=op)
        elapsed = time.perf_counter_ns() - r0
        timing.per_row_ns.append(elapsed)
        for cell in row.cells:
            if cell.value >= 0:
                grid.num_positive += 1
            else:
                grid.num_negative += 1
        grid.log.append(f"series={x}{sym} base_hash={row.base_hash} outs={len(row.cells)} elapsed_ns={elapsed}")
        grid.rows.append(row)
    timing.total_ns = time.perf_counter_ns() - t0
    return TimedRun(grid, timing)


def run_year_series(base_value: int, years: Sequence[int], store: Optional[HashBaseStore] = None) -> TimedRun:
    return run_series_op(MathOp.MUL, base_value, years, store=store)


def run_division_series(dividend: int, divisors: Sequence[int], store: Optional[HashBaseStore] = None) -> TimedRun:
    return run_series_op(MathOp.DIV, dividend, divisors, store=store)


def await_outs(grid: TimesPlankGrid) -> List[PlankOutput]:
    return [c for r in grid.rows for c in r.cells]


def select_depth(grid: TimesPlankGrid, depth: int) -> List[PlankOutput]:
    return [c for r in grid.rows[:depth] for c in r.cells]


def array_field(grid: TimesPlankGrid) -> List[List[int]]:
    return [[c.value for c in r.cells] for r in grid.rows]


def base_clock_outputs_per_sec(run: TimedRun) -> float:
    secs = run.timing.total_ns / 1e9
    return 0.0 if secs <= 0 else len(await_outs(run.grid)) / secs


def base_clock_hash_units_per_sec(run: TimedRun) -> float:
    secs = run.timing.total_ns / 1e9
    if secs <= 0:
        return 0.0
    units = sum(r.base_hash for r in run.grid.rows)
    return float(units) / secs


@dataclass
class CalcEvent:
    op: MathOp
    a: int
    b: int


@dataclass
class MathTreeNode:
    key_hash: int
    op: MathOp
    a: int
    b: int
    row: GridRow
    version: int = 1


@dataclass
class MathTree:
    nodes: Dict[int, MathTreeNode] = field(default_factory=dict)
    event_log: List[str] = field(default_factory=list)
    aspen_log: List[Dict[str, Any]] = field(default_factory=list)


def tree_key_hash(op: MathOp, a: int, b: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{op.value}|{a}|{b}".encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def process_math_event_loop(
    tree: MathTree,
    events: Sequence[CalcEvent],
    base_aspen: Optional[str] = None,
) -> None:
    """Event loop: address by hash, compare, and overwrite tree node when hash/output differs.
    If base_aspen is provided, each loop calculates by percentage and finds the aspen key
    corresponding to calculus status.
    """
    total = len(events)
    for i, ev in enumerate(events):
        key = tree_key_hash(ev.op, ev.a, ev.b)
        row = build_row_op(ev.op, ev.a, ev.b)
        sym = "x" if ev.op == MathOp.MUL else "/"
        node = tree.nodes.get(key)
        if node is None:
            action = "insert"
            tree.nodes[key] = MathTreeNode(key_hash=key, op=ev.op, a=ev.a, b=ev.b, row=row, version=1)
        else:
            if node.row.base_hash != row.base_hash or [(c.value, c.hash) for c in node.row.cells] != [(c.value, c.hash) for c in row.cells]:
                action = "overwrite"
                node.version += 1
                node.row = row
                node.op = ev.op
                node.a = ev.a
                node.b = ev.b
            else:
                action = "same"

        # percentage-based aspen lookup per loop iteration (calculus status)
        aspen_key = None
        if base_aspen:
            try:
                from .aspen import aspen_for_status, calculus_status
                st = calculus_status(action, i, total, op=ev.op.value, a=ev.a, b=ev.b, row_hash=row.base_hash)
                aspen_key = aspen_for_status(base_aspen, st)
                tree.aspen_log.append({"percentage": st["percentage_100"], "status": st, "aspen_key": aspen_key})
            except Exception:
                aspen_key = None

        suffix = f" aspen={aspen_key}" if aspen_key else ""
        if action == "insert":
            tree.event_log.append(f"insert key={key} {ev.a}{sym}{ev.b} hash={row.base_hash}{suffix}")
        elif action == "overwrite":
            tree.event_log.append(f"overwrite key={key} v={node.version} {ev.a}{sym}{ev.b} hash={row.base_hash}{suffix}")
        else:
            tree.event_log.append(f"same key={key} {ev.a}{sym}{ev.b} hash={row.base_hash} (no overwrite){suffix}")


def build_table_events_mul(base_value: int, n: int) -> List[CalcEvent]:
    return [CalcEvent(MathOp.MUL, base_value, i) for i in range(1, max(0, n) + 1)]




@dataclass
class SpeculativeCell:
    row: int
    col: int
    value: float
    hash: int
    category: str
    score: float


@dataclass
class SpeculativeMatrix:
    rows: int
    cols: int
    cells: List[SpeculativeCell] = field(default_factory=list)


def _hash_parts(parts: Sequence[str], value: float) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode())
        h.update(b"|")
    h.update(str(value).encode())
    return int.from_bytes(h.digest(), "little", signed=False)


def speculative_matrix_from_grid(grid: TimesPlankGrid, category: str = "math") -> SpeculativeMatrix:
    """Build speculative matrix from grid outputs (fundamental calcs -> drawn matrix interpretation)."""
    rows_n = len(grid.rows)
    cols_n = max((len(r.cells) for r in grid.rows), default=0)
    cells: List[SpeculativeCell] = []
    for i, row in enumerate(grid.rows):
        for j, out in enumerate(row.cells):
            val = float(out.value)
            cat = f"{category}:agent{out.agent_id}"
            h = _hash_parts([cat, f"{i}:{j}"], val)
            score = out.hash / float(2**64 - 1)
            cells.append(SpeculativeCell(i, j, val, h, cat, score))
    return SpeculativeMatrix(rows_n, cols_n, cells)


def cell_feature_vector(cell: SpeculativeCell) -> List[float]:
    return [cell.value, float(cell.hash % 1_000_000) / 1_000_000.0, cell.score, float(cell.row), float(cell.col)]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = na = nb = 0.0
    for i in range(min(len(a), len(b))):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def sort_by_category(matrix: SpeculativeMatrix, query_category: str, query_vector: Optional[Sequence[float]] = None) -> List[SpeculativeCell]:
    """Image-search-like category sort: rank by category match + similarity score."""
    ranked = list(matrix.cells)
    def key(c: SpeculativeCell):
        cat_match = 1.0 if query_category in c.category else 0.0
        sim = cosine_similarity(cell_feature_vector(c), query_vector) if query_vector is not None else c.score
        return (cat_match * 10.0 + sim, c.hash)
    ranked.sort(key=key, reverse=True)
    return ranked


@dataclass
class ConceptualPiece:
    id: int
    category: str
    weights: List[float]
    evidence_hash: int
    version: int = 1


class MathWeightModel:
    """Perfect internal nodes to new conceptual pieces and update math weights."""

    def __init__(self, learning_rate: float = 0.1):
        self.nodes: Dict[int, ConceptualPiece] = {}
        self.learning_rate = learning_rate
        self.log: List[str] = []

    def perfect_node(self, node_id: int, category: str, features: Sequence[float], evidence_hash: int) -> None:
        lr = self.learning_rate
        piece = self.nodes.get(node_id)
        if piece is None:
            self.nodes[node_id] = ConceptualPiece(node_id, category, list(features), evidence_hash, 1)
            self.log.append(f"insert node={node_id} cat={category} hash={evidence_hash}")
            return
        if piece.evidence_hash != evidence_hash or piece.category != category:
            n = max(len(piece.weights), len(features))
            if len(piece.weights) < n:
                piece.weights.extend([0.0] * (n - len(piece.weights)))
            for i, f in enumerate(features):
                piece.weights[i] = piece.weights[i] * (1.0 - lr) + float(f) * lr
            piece.category = category
            piece.evidence_hash = evidence_hash
            piece.version += 1
            self.log.append(f"update node={node_id} v={piece.version} cat={category} hash={evidence_hash}")
        else:
            self.log.append(f"same node={node_id} hash={evidence_hash} (no update)")


def refine_model_from_matrix(model: MathWeightModel, matrix: SpeculativeMatrix, query_category: str) -> List[SpeculativeCell]:
    """From speculative matrix + category sort, perfect model nodes and update weights."""
    ranked = sort_by_category(matrix, query_category, None)
    for i, cell in enumerate(ranked):
        node_id = _hash_parts([cell.category, str(i)], cell.value)
        feats = cell_feature_vector(cell)
        model.perfect_node(node_id, cell.category, feats, cell.hash)
    return ranked


class CategorySearchAdapter:
    """Adapter interface to connect external category-search algorithms (e.g. image-search-like APIs).
    Implement `search(category, features, limit)` to integrate providers (e.g. Google image/category search)
    and feed results back into `refine_model_from_matrix` / conceptual pieces.
    """
    def search(self, category: str, features: Sequence[float], limit: int = 10) -> List[Dict[str, Any]]:
        return []  # no external provider configured


def integrate_category_search(model: MathWeightModel, matrix: SpeculativeMatrix, category: str, adapter: Optional[CategorySearchAdapter] = None) -> Dict[str, Any]:
    """Local category-sort + optional external adapter; updates model internal nodes/weights."""
    ranked = refine_model_from_matrix(model, matrix, category)
    external: List[Dict[str, Any]] = []
    if adapter is not None and ranked:
        qv = cell_feature_vector(ranked[0])
        external = adapter.search(category, qv, limit=min(10, len(ranked)))
    return {"ranked_count": len(ranked), "external_results": external, "model_nodes": len(model.nodes), "log_tail": model.log[-5:]}

# Add-ons: mathematical shortcuts (shortened hashes and arithmetic values)
from .shortcuts import (
    ShortAddons,
    b62_decode,
    b62_encode,
    compress_arith_value,
    pack_u64,
    shorten_hash,
    shortcut_div_small,
    shortcut_mul_small,
    simplify_log,
    unpack_u64,
)

# Trace embeds: forward-null/back-null aware statements
from .trace_embed import (
    TraceEmbed,
    TraceEntry,
    embed_trace,
    safe_execute_statement,
    statement_with_trace,
)
