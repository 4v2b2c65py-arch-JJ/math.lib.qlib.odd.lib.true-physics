"""Read/check benchmark.gguf, compare per required year, update values, and
produce timeline hash/timestamp against current machine time.

GGUF layout produced by src/bin/bench.rs:
- magic b"GGUF", version u32=3, tensor_count u64, kv_count u64
- KV entries: key(string), type(u32), value
  types used: 4=UINT64, 8=STRING, 11=INT64, 12=FLOAT64
- trailing blob: per-output records (60 bytes each):
  year:i64, agent_id:u32, addr_x:i64, addr_y:i64, addr_z:i64,
  value:i64, hash:u64, elapsed_ns:u64 (all little-endian)
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import Address, HashBaseStore, PlankOutput, base, base_clock_hash_units_per_sec, base_clock_outputs_per_sec, run_year_series

GGUF_MAGIC = b"GGUF"
REC_FMT = "<q i q q q q Q Q"  # year, agent_id(u32 but struct i ok via pack), x,y,z, value, hash, elapsed_ns
REC_SIZE = struct.calcsize("<q I q q q q Q Q")


@dataclass
class GGUFData:
    version: int
    tensor_count: int
    kv: Dict[str, Any]
    records: List[dict]


def _read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_i64(f) -> int:
    return struct.unpack("<q", f.read(8))[0]


def _read_f64(f) -> float:
    return struct.unpack("<d", f.read(8))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8")


def read_benchmark_gguf(path: Path) -> GGUFData:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not GGUF: {magic!r}")
        version = _read_u32(f)
        tensor_count = _read_u64(f)
        kv_count = _read_u64(f)
        kv: Dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_str(f)
            vtype = _read_u32(f)
            if vtype == 4:
                val = _read_u64(f)
            elif vtype == 8:
                val = _read_str(f)
            elif vtype == 10:
                val = _read_u32(f)
            elif vtype == 11:
                val = _read_i64(f)
            elif vtype == 12:
                val = _read_f64(f)
            else:
                raise ValueError(f"unsupported kv type {vtype} for {key}")
            kv[key] = val
        blob_bytes = int(kv.get("mathlib.blob_bytes", 0))
        blob = f.read(blob_bytes) if blob_bytes else f.read()
    records = []
    off = 0
    fmt = "<q I q q q q Q Q"
    size = struct.calcsize(fmt)
    while off + size <= len(blob):
        year, agent_id, ax, ay, az, value, h, elapsed_ns = struct.unpack_from(fmt, blob, off)
        records.append(
            {
                "year": year,
                "agent_id": agent_id,
                "address": (ax, ay, az),
                "value": value,
                "hash": h,
                "elapsed_ns": elapsed_ns,
            }
        )
        off += size
    return GGUFData(version=version, tensor_count=tensor_count, kv=kv, records=records)


def records_by_year(data: GGUFData) -> Dict[int, List[dict]]:
    by: Dict[int, List[dict]] = {}
    for r in data.records:
        by.setdefault(int(r["year"]), []).append(r)
    return by


def _current_machine_timestamps() -> dict:
    return {
        "time_ns": time.time_ns(),
        "perf_ns": time.perf_counter_ns(),
        "mono_ns": time.monotonic_ns(),
    }


def _timeline_hash(machine_ts_ns: int, year_x: int, base_hash: int, outputs: Sequence[PlankOutput]) -> Tuple[str, int]:
    """Reformat calculation into our formula producing a new machine timestamp in hash."""
    h = hashlib.blake2b(digest_size=32)
    h.update(str(machine_ts_ns).encode())
    h.update(b"|")
    h.update(str(year_x).encode())
    h.update(b"|")
    h.update(str(base_hash).encode())
    h.update(b"|")
    for o in outputs:
        h.update(f"{o.agent_id}:{o.value}:{o.hash}:{o.address.x},{o.address.y},{o.address.z};".encode())
    digest = h.digest()
    hex_hash = digest.hex()
    # derive a timeline timestamp value (64-bit) from hash mixed with machine ts and year
    ts64 = int.from_bytes(digest[:8], "little", signed=False) ^ (machine_ts_ns & 0xFFFFFFFFFFFFFFFF) ^ (year_x & 0xFFFFFFFF)
    return hex_hash, ts64


def compare_and_update_year(
    data: GGUFData,
    base_value: int,
    year: int,
    store: Optional[HashBaseStore] = None,
) -> dict:
    """Check GGUF for required year, compare against current calc, update val, return comparison."""
    by = records_by_year(data)
    stored = by.get(int(year), [])
    row = base(base_value, int(year), store=store)
    current = [
        {
            "year": int(year),
            "agent_id": c.agent_id,
            "address": (c.address.x, c.address.y, c.address.z),
            "value": c.value,
            "hash": c.hash,
        }
        for c in row.cells
    ]

    # Compare by (agent_id,address)
    def keyer(r):
        return (int(r["agent_id"]), tuple(r["address"]))

    stored_map = {keyer(r): r for r in stored}
    current_map = {keyer(r): r for r in current}
    diffs = []
    for k, cur in current_map.items():
        st = stored_map.get(k)
        if st is None:
            diffs.append({"key": k, "status": "missing_in_gguf", "current": cur})
        elif int(st["value"]) != int(cur["value"]) or int(st["hash"]) != int(cur["hash"]):
            diffs.append({"key": k, "status": "mismatch", "stored": {"value": st["value"], "hash": st["hash"]}, "current": cur})
            # update stored record value/hash in-memory
            st["value"] = cur["value"]
            st["hash"] = cur["hash"]
        else:
            diffs.append({"key": k, "status": "match", "value": cur["value"]})

    machine = _current_machine_timestamps()
    timeline_hex, timeline_ts = _timeline_hash(machine["time_ns"], int(year), row.base_hash, row.cells)

    return {
        "year_x": int(year),
        "base": base_value,
        "base_hash": row.base_hash,
        "stored_count": len(stored),
        "current_count": len(current),
        "diffs": diffs,
        "machine_timestamps": machine,
        "timeline_hash": timeline_hex,
        "timeline_timestamp": timeline_ts,
        "outputs": current,
    }


def run_timeline_check(
    gguf_path: Path,
    required_years: Sequence[int],
    base_value: Optional[int] = None,
    update_out_path: Optional[Path] = None,
) -> dict:
    """Read GGUF, run comparison/update per required year, optionally write updated GGUF, return results."""
    data = read_benchmark_gguf(gguf_path)
    b = int(base_value if base_value is not None else data.kv.get("mathlib.base", 8))
    store = HashBaseStore(base_num=b)
    results = []
    for y in required_years:
        results.append(compare_and_update_year(data, b, int(y), store=store))
    out = {
        "gguf": str(gguf_path),
        "base": b,
        "store_summary": store.summary(),
        "results": results,
        "clock_outputs_per_sec": base_clock_outputs_per_sec(run_year_series(b, required_years)),
        "clock_hash_units_per_sec": base_clock_hash_units_per_sec(run_year_series(b, required_years)),
    }
    if update_out_path is not None:
        write_updated_gguf(data, update_out_path)
        out["updated_gguf"] = str(update_out_path)
    return out


def write_updated_gguf(data: GGUFData, path: Path) -> None:
    """Write GGUF with updated KV summary and updated records blob."""
    # rebuild blob from records
    blob = bytearray()
    for r in data.records:
        blob.extend(struct.pack("<q", int(r["year"])))
        blob.extend(struct.pack("<I", int(r["agent_id"])))
        ax, ay, az = r["address"]
        blob.extend(struct.pack("<q q q", int(ax), int(ay), int(az)))
        blob.extend(struct.pack("<q", int(r["value"])))
        blob.extend(struct.pack("<Q", int(r["hash"]) & 0xFFFFFFFFFFFFFFFF))
        blob.extend(struct.pack("<Q", int(r.get("elapsed_ns", 0)) & 0xFFFFFFFFFFFFFFFF))

    kv = dict(data.kv)
    kv["mathlib.blob_bytes"] = len(blob)
    kv["mathlib.outputs"] = len(data.records)

    def write_u32(f, v): f.write(struct.pack("<I", int(v) & 0xFFFFFFFF))
    def write_u64(f, v): f.write(struct.pack("<Q", int(v) & 0xFFFFFFFFFFFFFFFF))
    def write_i64(f, v): f.write(struct.pack("<q", int(v)))
    def write_f64(f, v): f.write(struct.pack("<d", float(v)))
    def write_str(f, s):
        b = s.encode("utf-8")
        write_u64(f, len(b)); f.write(b)
    def write_kv(f, k, v):
        write_str(f, k)
        if isinstance(v, str):
            write_u32(f, 8); write_str(f, v)
        elif isinstance(v, float):
            write_u32(f, 12); write_f64(f, v)
        elif isinstance(v, int) and not isinstance(v, bool):
            # heuristic: negative -> int64 else uint64 except known int64 keys
            if v < 0 or k in ("mathlib.base",):
                write_u32(f, 11); write_i64(f, v)
            else:
                write_u32(f, 4); write_u64(f, v)
        else:
            write_u32(f, 8); write_str(f, str(v))

    with open(path, "wb") as f:
        f.write(GGUF_MAGIC)
        write_u32(f, 3)
        write_u64(f, 0)
        write_u64(f, len(kv))
        for k, v in kv.items():
            write_kv(f, k, v)
        f.write(blob)
