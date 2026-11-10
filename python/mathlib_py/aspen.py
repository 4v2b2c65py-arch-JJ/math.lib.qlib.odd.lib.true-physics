"""Aspen key: percentage-based selection per calculation loop, mapped to calculus status.

"Use aspen by percentage so every time a new loop calculate it find the aspen
corresponding to calculus status."

Authority in domain structure: recall returns a percentage computable in field theory.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _hex_digest(material: str, digest_size: int = 16) -> str:
    return hashlib.blake2b(material.encode("utf-8"), digest_size=digest_size).hexdigest()


def derive_base_aspen_key(
    phrase_hash: str,
    seed_hash: str = "",
    tree_nodes: int = 0,
    series: Optional[Sequence[int]] = None,
) -> str:
    """Deterministic base aspen key from seed material."""
    series_s = ",".join(str(x) for x in (series or []))
    material = "|".join(["aspen", str(phrase_hash), str(seed_hash), str(tree_nodes), series_s])
    return _hex_digest(material, digest_size=16)


def calculus_status(action: str, index: int, total: int, **extra: Any) -> Dict[str, Any]:
    """Describe calculus status for a loop iteration."""
    total_n = max(1, int(total))
    idx = max(0, int(index))
    pct = (idx / (total_n - 1)) if total_n > 1 else 1.0
    pct = max(0.0, min(1.0, pct))
    return {
        "action": str(action),
        "index": idx,
        "total": total_n,
        "percentage": pct,
        "percentage_100": round(pct * 100.0, 6),
        **extra,
    }


@dataclass
class AspenRecord:
    percentage: float
    status: Dict[str, Any]
    aspen_key: str


@dataclass
class AspenLedger:
    base_aspen: str
    records: List[AspenRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_aspen": self.base_aspen,
            "count": len(self.records),
            "records": [
                {"percentage": r.percentage, "status": r.status, "aspen_key": r.aspen_key}
                for r in self.records
            ],
        }


def aspen_for_status(base_aspen: str, status: Dict[str, Any]) -> str:
    """Find/derive the aspen key corresponding to calculus status (percentage-based per loop)."""
    pct = float(status.get("percentage", 0.0))
    bucket = int(round(pct * 10000))  # basis points
    material = "|".join(
        [
            str(base_aspen),
            str(bucket),
            f"{pct:.12f}",
            str(status.get("action", "")),
            str(status.get("index", "")),
            str(status.get("total", "")),
            str(status.get("op", "")),
            str(status.get("a", "")),
            str(status.get("b", "")),
            str(status.get("row_hash", "")),
        ]
    )
    return _hex_digest(material, digest_size=16)


def aspen_for_percentage(base_aspen: str, percentage: float, salt: str = "") -> str:
    pct = max(0.0, min(1.0, float(percentage)))
    bucket = int(round(pct * 10000))
    material = f"{base_aspen}|{bucket}|{pct:.12f}|{salt}"
    return _hex_digest(material, digest_size=16)


@dataclass
class SpanTickBucket:
    """Bucket spanning a tick/percentage range; open buckets collect aspen keys."""
    index: int
    start_pct: float
    end_pct: float
    tick_start: float
    tick_end: float
    open: bool = True
    aspen_keys: List[str] = field(default_factory=list)
    statuses: List[Dict[str, Any]] = field(default_factory=list)

    def contains(self, pct: float, tick: Optional[float] = None) -> bool:
        in_pct = self.start_pct <= pct <= self.end_pct or (self.index == 0 and pct < self.start_pct)
        if tick is None:
            return in_pct
        lo, hi = sorted([self.tick_start, self.tick_end])
        return in_pct and (lo <= tick <= hi)


@dataclass
class BucketList:
    """Bucket list spanning ticks/percentages for aspen assignment."""
    base_aspen: str
    buckets: List[SpanTickBucket] = field(default_factory=list)
    open_bucket_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_span(cls, base_aspen: str, n: int = 10, tick_min: float = 0.0, tick_max: float = 1.0) -> "BucketList":
        n = max(1, int(n))
        buckets: List[SpanTickBucket] = []
        for i in range(n):
            start_pct = i / n
            end_pct = (i + 1) / n if i < n - 1 else 1.0
            t0 = tick_min + (tick_max - tick_min) * start_pct
            t1 = tick_min + (tick_max - tick_min) * end_pct
            buckets.append(SpanTickBucket(index=i, start_pct=start_pct, end_pct=end_pct, tick_start=t0, tick_end=t1, open=True))
        return cls(base_aspen=base_aspen, buckets=buckets, open_bucket_indices=list(range(n)))

    def open_buckets(self) -> List[SpanTickBucket]:
        if self.open_bucket_indices:
            idx = set(self.open_bucket_indices)
            return [b for b in self.buckets if b.index in idx and b.open]
        return [b for b in self.buckets if b.open]

    def add_aspen_to_open(self, aspen_key: str, percentage: float, tick: Optional[float] = None, status: Optional[Dict[str, Any]] = None) -> Optional[int]:
        pct = max(0.0, min(1.0, float(percentage)))
        for b in self.open_buckets():
            if b.contains(pct, tick=tick):
                if aspen_key not in b.aspen_keys:
                    b.aspen_keys.append(aspen_key)
                if status is not None:
                    b.statuses.append(status)
                return b.index
        # fallback: nearest open bucket by percentage
        opens = self.open_buckets()
        if not opens:
            return None
        b = min(opens, key=lambda x: abs(((x.start_pct + x.end_pct) / 2.0) - pct))
        if aspen_key not in b.aspen_keys:
            b.aspen_keys.append(aspen_key)
        if status is not None:
            b.statuses.append(status)
        return b.index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_aspen": self.base_aspen,
            "open_bucket_indices": self.open_bucket_indices,
            "buckets": [
                {
                    "index": b.index,
                    "start_pct": b.start_pct,
                    "end_pct": b.end_pct,
                    "tick_start": b.tick_start,
                    "tick_end": b.tick_end,
                    "open": b.open,
                    "aspen_keys": b.aspen_keys,
                    "count": len(b.aspen_keys),
                }
                for b in self.buckets
            ],
        }

@dataclass
class DomainAuthority:
    """Authority entry in a domain structure; recall yields computable field percentage."""
    domain: str
    aspen_key: str
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


class DomainStructure:
    """Domain authority registry for fluid recall (authority-in-domain)."""

    def __init__(self):
        self.domains: Dict[str, List[DomainAuthority]] = {}

    def register(self, domain: str, aspen_key: str, weight: float = 1.0, meta: Optional[Dict[str, Any]] = None) -> DomainAuthority:
        auth = DomainAuthority(domain=str(domain), aspen_key=str(aspen_key), weight=float(weight), meta=meta or {})
        self.domains.setdefault(auth.domain, []).append(auth)
        return auth

    def authorities(self, domain: str) -> List[DomainAuthority]:
        return list(self.domains.get(str(domain), []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": {
                k: [{"aspen_key": a.aspen_key, "weight": a.weight, "meta": a.meta} for a in v]
                for k, v in self.domains.items()
            }
        }


def field_percentage(base_aspen: str, domain: str, status: Dict[str, Any]) -> float:
    """Compute a field-theory percentage from aspen+domain+calculus status (deterministic, in [0,1])."""
    status_pct = max(0.0, min(1.0, float(status.get("percentage", 0.0))))
    h = hashlib.blake2b(
        f"field|{base_aspen}|{domain}|{status_pct:.12f}|{status.get('action','')}|{status.get('op','')}|{status.get('row_hash','')}".encode("utf-8"),
        digest_size=8,
    ).digest()
    u = int.from_bytes(h, "little") / float(2**64 - 1)
    # blend trajectory percentage with field component for computable recall
    field_pct = (0.618 * status_pct + 0.382 * u)
    return max(0.0, min(1.0, field_pct))


def recall_aspen_in_domain(
    domain_structure: DomainStructure,
    base_aspen: str,
    domain: str,
    status: Dict[str, Any],
) -> Dict[str, Any]:
    """Recall using authority in domain structure: returns aspen and percentage computable in field theory."""
    st = dict(status)
    st_pct = max(0.0, min(1.0, float(st.get("percentage", 0.0))))
    # if caller gave 0..100
    if st_pct > 1.0:
        st_pct = st_pct / 100.0
    st["percentage"] = st_pct
    if "percentage_100" not in st:
        st["percentage_100"] = round(st_pct * 100.0, 6)

    aspen_key = aspen_for_status(base_aspen, st)
    f_pct = field_percentage(base_aspen, domain, st)
    auths = domain_structure.authorities(domain)
    # register authority (fluid: first/strongest match in domain context)
    if not any(a.aspen_key == aspen_key for a in auths):
        domain_structure.register(domain, aspen_key, weight=f_pct, meta={"status_action": st.get("action"), "op": st.get("op")})
        auths = domain_structure.authorities(domain)

    authority = next((a for a in auths if a.aspen_key == aspen_key), auths[0] if auths else None)
    return {
        "domain": str(domain),
        "calculus_status": st,
        "percentage": st_pct,
        "percentage_100": st["percentage_100"],
        "field_percentage": f_pct,
        "field_percentage_100": round(f_pct * 100.0, 6),
        "aspen_key": aspen_key,
        "authority": {"aspen_key": authority.aspen_key, "weight": authority.weight} if authority else None,
    }
