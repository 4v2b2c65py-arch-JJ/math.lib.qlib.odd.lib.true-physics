"""Trace embeds for forward-null/back-null so statements can return data before hitting null status.

Provides trace records, embedded trace markers, and a safe statement executor that
scans forward/backward around null boundaries and returns available pre-null data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class TraceEntry:
    index: int
    direction: str  # "forward" | "backward" | "origin"
    offset: int
    value: Any
    is_null: bool
    key: Optional[str] = None


@dataclass
class TraceEmbed:
    """Embedded trace context accounting for forward-null and back-null boundaries."""
    origin_index: int
    total: int
    forward_null_index: Optional[int] = None  # first null at/after origin
    back_null_index: Optional[int] = None     # first null at/before origin
    pre_null_forward: List[TraceEntry] = field(default_factory=list)  # data before forward null
    pre_null_backward: List[TraceEntry] = field(default_factory=list) # data before back null (toward origin)
    trace: List[TraceEntry] = field(default_factory=list)
    status: str = "ok"  # ok | forward_null | back_null | both_null | empty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_index": self.origin_index,
            "total": self.total,
            "forward_null_index": self.forward_null_index,
            "back_null_index": self.back_null_index,
            "status": self.status,
            "pre_null_forward": [e.__dict__ for e in self.pre_null_forward],
            "pre_null_backward": [e.__dict__ for e in self.pre_null_backward],
            "trace": [e.__dict__ for e in self.trace],
        }


def _is_null(v: Any) -> bool:
    return v is None


def embed_trace(seq: Sequence[Any], origin_index: Optional[int] = None, *, window: int = 8) -> TraceEmbed:
    """Build trace embed around origin accounting for forward-null and back-null.
    Captures available data before hitting null status in each direction.
    """
    n = len(seq)
    if n == 0:
        return TraceEmbed(origin_index=-1, total=0, status="empty")
    o = n - 1 if origin_index is None else max(0, min(int(origin_index), n - 1))

    te = TraceEmbed(origin_index=o, total=n)

    # forward scan from origin -> find first null and collect pre-null data
    pre_fwd: List[TraceEntry] = []
    for i in range(o, n):
        v = seq[i]
        off = i - o
        if _is_null(v):
            te.forward_null_index = i
            break
        pre_fwd.append(TraceEntry(index=i, direction="forward", offset=off, value=v, is_null=False))
        te.trace.append(TraceEntry(index=i, direction="forward", offset=off, value=v, is_null=False))
    te.pre_null_forward = pre_fwd

    # backward scan from origin -> 0 find first null and collect pre-null data (toward origin)
    pre_back: List[TraceEntry] = []
    for i in range(o, -1, -1):
        v = seq[i]
        off = o - i
        if _is_null(v):
            te.back_null_index = i
            break
        pre_back.append(TraceEntry(index=i, direction="backward", offset=off, value=v, is_null=False))
        if i != o:  # avoid duplicating origin in trace
            te.trace.append(TraceEntry(index=i, direction="backward", offset=off, value=v, is_null=False))
    te.pre_null_backward = pre_back

    if te.forward_null_index is not None and te.back_null_index is not None:
        te.status = "both_null"
    elif te.forward_null_index is not None:
        te.status = "forward_null"
    elif te.back_null_index is not None:
        te.status = "back_null"
    else:
        te.status = "ok"
    return te


def statement_with_trace(seq: Sequence[Any], origin_index: Optional[int] = None) -> Dict[str, Any]:
    """Return statement/data that can surface results before hitting null status.
    Prefers pre-null-forward, falls back to pre-null-backward, else empty.
    """
    emb = embed_trace(seq, origin_index=origin_index)
    # data available before null
    data = [e.value for e in emb.pre_null_forward] or [e.value for e in reversed(emb.pre_null_backward)]
    returned = data if data else []
    return {
        "status": emb.status,
        "hit_forward_null": emb.forward_null_index is not None,
        "hit_back_null": emb.back_null_index is not None,
        "returned_before_null": returned,
        "count": len(returned),
        "trace_embed": emb.to_dict(),
    }


def safe_execute_statement(
    fn: Callable[..., Any],
    *args: Any,
    fallback: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute statement; on exception/null-like failure return trace-aware result with fallback data."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        emb = embed_trace([fallback] if fallback is not None else [], origin_index=0)
        emb.status = "error"
        return {
            "ok": False,
            "error": str(e),
            "returned_before_null": [fallback] if fallback is not None else [],
            "trace_embed": emb.to_dict(),
        }
    if _is_null(result):
        # null status: return fallback/pre-null data if provided
        buf = [fallback] if fallback is not None else []
        emb = embed_trace([*buf, None], origin_index=0 if buf else 0)
        return {
            "ok": False,
            "null_status": True,
            "returned_before_null": [x for x in buf if not _is_null(x)],
            "trace_embed": emb.to_dict(),
        }
    return {"ok": True, "result": result, "returned_before_null": [result], "trace_embed": embed_trace([result], 0).to_dict()}
