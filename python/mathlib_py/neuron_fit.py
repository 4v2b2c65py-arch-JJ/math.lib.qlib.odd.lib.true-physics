"""Neuron fitting over qbit structures with log-sample compression.

Uses simplify_log (log samples/magnitude decomposition) to compress network
weights and qbit-derived feature vectors. Includes self-updating compute gauge
and self-improving notes driven by throughput/net gains.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def qbit_to_features(qbit: Any, value: Optional[float] = None) -> List[float]:
    """Extract a neuron input feature vector from a QbitStructure-like object."""
    alpha = float(getattr(qbit, "alpha", 0.0))
    beta = float(getattr(qbit, "beta", 0.0))
    phase = float(getattr(qbit, "phase", 0.0))
    feats = [alpha, beta, math.sin(phase), math.cos(phase), alpha * alpha - beta * beta]
    if value is not None:
        try:
            vf = float(value)
            feats.append(math.tanh(vf) if abs(vf) < 1e6 else (1.0 if vf > 0 else -1.0))
        except Exception:
            feats.append(0.0)
    return feats


def log_sample_features(values: Sequence[float], base: float = 10.0) -> List[Dict[str, Any]]:
    """Generate log samples for a sequence of magnitudes (for compression)."""
    from .shortcuts import simplify_log
    samples = []
    for v in values:
        try:
            vf = float(v)
        except Exception:
            samples.append({"x": v, "ok": False})
            continue
        av = abs(vf)
        if av == 0.0:
            samples.append({"x": vf, "ok": True, "zero": True, "value": None})
            continue
        try:
            s = simplify_log(av, base=base)
            s["sign"] = -1 if vf < 0 else 1
            s["ok"] = True
            samples.append(s)
        except Exception as e:
            samples.append({"x": vf, "ok": False, "error": str(e)})
    return samples


def compress_vector_log(weights: Sequence[float], base: float = 10.0, mantissa_bits: int = 8) -> Dict[str, Any]:
    """Compress a weight/feature vector using log samples (magnitude decomposition)."""
    from .shortcuts import simplify_log
    entries = []
    q_levels = max(2, (1 << max(1, int(mantissa_bits))) - 1)
    for w in weights:
        wf = float(w)
        if wf == 0.0:
            entries.append({"z": True})
            continue
        sign = -1 if wf < 0 else 1
        aw = abs(wf)
        try:
            s = simplify_log(aw, base=base)
            exp10 = int(s.get("exp10", 0))
            mant = float(s.get("mantissa", 1.0))
            if base == 10.0:
                q = int(round(((mant - 1.0) / 9.0) * q_levels))
            else:
                q = int(round(mant * q_levels))
            q = max(0, min(q_levels, q))
            entries.append({"s": sign, "e": exp10, "m": q})
        except Exception:
            entries.append({"s": sign, "raw": wf})
    packed = []
    for e in entries:
        if e.get("z"):
            packed.append("0")
        elif "raw" in e:
            packed.append(f"r{e['s']}:{e['raw']}")
        else:
            packed.append(f"{e['s']}:{e['e']}:{e['m']}")
    blob = ";".join(packed)
    return {
        "base": base,
        "mantissa_bits": mantissa_bits,
        "n": len(weights),
        "entries": entries,
        "blob": blob,
        "blob_len": len(blob),
        "orig_len_est": len(",".join(str(float(w)) for w in weights)),
    }


def decompress_vector_log(compressed: Dict[str, Any]) -> List[float]:
    """Approximate reconstruction from log-sample compressed vector."""
    base = float(compressed.get("base", 10.0))
    mantissa_bits = int(compressed.get("mantissa_bits", 8))
    q_levels = max(2, (1 << max(1, mantissa_bits)) - 1)
    out: List[float] = []
    for e in compressed.get("entries", []):
        if e.get("z"):
            out.append(0.0)
            continue
        if "raw" in e:
            out.append(float(e["raw"]))
            continue
        sign = int(e.get("s", 1))
        exp = int(e.get("e", 0))
        q = int(e.get("m", 0))
        if base == 10.0:
            mant = 1.0 + 9.0 * (q / q_levels)
            val = sign * mant * (10.0 ** exp)
        else:
            mant = q / q_levels
            val = sign * (base ** exp) * max(mant, 1e-12)
        out.append(float(val))
    return out


@dataclass
class Neuron:
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    lr: float = 0.05

    def predict(self, x: Sequence[float]) -> float:
        n = min(len(self.weights), len(x))
        s = self.bias
        for i in range(n):
            s += self.weights[i] * float(x[i])
        return s

    def fit(self, xs: Sequence[Sequence[float]], ys: Sequence[float], epochs: int = 5) -> Dict[str, Any]:
        xs_list = [list(map(float, x)) for x in xs]
        ys_list = [float(y) for y in ys]
        if not xs_list:
            return {"samples": 0, "epochs": 0, "ops": 0, "elapsed_ns": 0}
        dim = max(len(x) for x in xs_list)
        if len(self.weights) < dim:
            self.weights.extend([0.0] * (dim - len(self.weights)))
        epochs_n = max(1, int(epochs))
        ops = 0
        t0 = time.perf_counter_ns()
        for _ in range(epochs_n):
            for x, y in zip(xs_list, ys_list):
                pred = self.predict(x)
                err = y - pred
                for i, xi in enumerate(x):
                    self.weights[i] += self.lr * err * float(xi)
                self.bias += self.lr * err
                ops += len(x) + 1
        t1 = time.perf_counter_ns()
        return {"samples": len(xs_list), "epochs": epochs_n, "ops": ops, "elapsed_ns": max(1, t1 - t0), "dim": dim}

    def fit_from_qbits(self, qbits: Sequence[Any], targets: Sequence[float], values: Optional[Sequence[float]] = None, epochs: int = 5) -> Dict[str, Any]:
        xs = []
        for i, qb in enumerate(qbits):
            v = None if values is None or i >= len(values) else values[i]
            xs.append(qbit_to_features(qb, v))
        return self.fit(xs, targets, epochs=epochs)

    def compress(self, base: float = 10.0, mantissa_bits: int = 8) -> Dict[str, Any]:
        return {
            "weights": compress_vector_log(self.weights, base=base, mantissa_bits=mantissa_bits),
            "bias": compress_vector_log([self.bias], base=base, mantissa_bits=mantissa_bits),
        }


@dataclass
class ComputeGauge:
    """Self-updating compute gauge: tracks throughput and percentage against baseline."""
    baseline_ops_per_sec: float = 0.0
    ops_per_sec: float = 0.0
    throughput_percentage: float = 0.0
    above_30: bool = False
    samples: int = 0
    ops: int = 0
    elapsed_ns: int = 0
    updates: int = 0

    def update_self(self, samples: int, ops: int, elapsed_ns: int, baseline_ops_per_sec: Optional[float] = None) -> Dict[str, Any]:
        self.samples = int(samples)
        self.ops = int(ops)
        self.elapsed_ns = max(1, int(elapsed_ns))
        elapsed_s = self.elapsed_ns / 1e9
        self.ops_per_sec = self.ops / elapsed_s if elapsed_s > 0 else 0.0
        if baseline_ops_per_sec is not None and baseline_ops_per_sec > 0:
            self.baseline_ops_per_sec = float(baseline_ops_per_sec)
        elif self.baseline_ops_per_sec <= 0:
            # bootstrap baseline so first run establishes reference (100%)
            self.baseline_ops_per_sec = self.ops_per_sec if self.ops_per_sec > 0 else 1.0
        self.throughput_percentage = (self.ops_per_sec / self.baseline_ops_per_sec) * 100.0 if self.baseline_ops_per_sec > 0 else 0.0
        self.above_30 = self.throughput_percentage > 30.0
        self.updates += 1
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_ops_per_sec": self.baseline_ops_per_sec,
            "ops_per_sec": self.ops_per_sec,
            "throughput_percentage": self.throughput_percentage,
            "above_30": self.above_30,
            "samples": self.samples,
            "ops": self.ops,
            "elapsed_ns": self.elapsed_ns,
            "updates": self.updates,
        }


def self_improving_notes(gauge: Dict[str, Any], net_gain: Dict[str, Any], previous_notes: Optional[List[Dict[str, Any]]] = None,
                         goal: str = "foreseeable result that future holds") -> Dict[str, Any]:
    """Recursively improve notes when net gains and throughput indicate progress toward goal."""
    prev = list(previous_notes or [])
    pct = float(gauge.get("throughput_percentage", 0.0) or 0.0)
    above = bool(gauge.get("above_30", False)) or pct > 30.0
    gain_ok = bool(net_gain.get("improved", False)) or float(net_gain.get("gain_ratio", 0.0) or 0.0) >= 0.0
    foreseeable = above and gain_ok
    level = len(prev)
    note = {
        "level": level,
        "timestamp_ns": time.time_ns(),
        "throughput_percentage": pct,
        "above_30": above,
        "net_gain": net_gain,
        "foreseeable_result": foreseeable,
        "goal": goal,
        "summary": (
            f"Compute gauge {pct:.2f}% ({'above' if above else 'at/below'} 30). "
            f"Net gain {'positive' if gain_ok else 'negative'}; "
            f"goal '{goal}' is {'foreseeable' if foreseeable else 'not yet foreseeable'}."
        ),
    }
    # self-recursive improvement: refine prior note based on current gain when progress holds
    if prev and foreseeable:
        last = dict(prev[-1])
        last_refined = {
            "level": last.get("level", level - 1),
            "refined_at_level": level,
            "timestamp_ns": time.time_ns(),
            "refined_summary": f"[refined@{level}] {last.get('summary','')} -> throughput {pct:.2f}%, net gain sustained.",
            "throughput_percentage": pct,
            "foreseeable_result": True,
        }
        prev[-1] = last_refined
    notes = prev + [note]
    return {
        "current": note,
        "notes": notes,
        "count": len(notes),
        "recursively_improved": bool(prev and foreseeable),
        "foreseeable_result": foreseeable,
        "above_30": above,
    }


def compress_qbits(qbits: Sequence[Any], values: Optional[Sequence[float]] = None, base: float = 10.0) -> Dict[str, Any]:
    """Compress qbit states (and optional values) into log-sample feature representation."""
    mats = []
    for i, qb in enumerate(qbits):
        v = None if values is None or i >= len(values) else values[i]
        feats = qbit_to_features(qb, v)
        mats.append({
            "features": feats,
            "log_samples": log_sample_features([abs(f) if f != 0 else 0.0 for f in feats], base=base),
            "compressed": compress_vector_log(feats, base=base),
        })
    return {"count": len(mats), "base": base, "qbits": mats}


def fit_and_compress_net(qbits: Sequence[Any], targets: Sequence[float], values: Optional[Sequence[float]] = None,
                         base: float = 10.0, epochs: int = 8, mantissa_bits: int = 8,
                         state_path: Optional[str] = "data/neuron_fit_state.json",
                         goal: str = "foreseeable result that future holds") -> Dict[str, Any]:
    """Fit a neuron on qbit features and compress the resulting net with new log samples.
    Updates compute gauge (self-updating) and self-improving notes when throughput > 30% and net gains are foreseeable.
    """
    neuron = Neuron(lr=0.05)
    fit_stats = neuron.fit_from_qbits(qbits, targets, values=values, epochs=epochs)

    # use new log samples from weights/targets to drive compression
    w_mags = [abs(w) for w in neuron.weights if w != 0.0] or [0.0]
    t_mags = [abs(float(t)) for t in targets if float(t) != 0.0] or [0.0]
    weight_log_samples = log_sample_features(w_mags, base=base)
    target_log_samples = log_sample_features(t_mags[:64], base=base)
    compressed = neuron.compress(base=base, mantissa_bits=mantissa_bits)
    recon = decompress_vector_log(compressed["weights"])

    # rough net gain signal (compression ratio and reconstruction closeness)
    orig = max(1, int(compressed["weights"].get("orig_len_est", 1)))
    comp_len = max(1, int(compressed["weights"].get("blob_len", 1)))
    compression_ratio = orig / comp_len
    # simple gain proxy: better when compressed and gauge healthy
    net_gain = {
        "compression_ratio": compression_ratio,
        "gain_ratio": compression_ratio - 1.0,
        "improved": compression_ratio >= 1.0,
    }

    # self-updating compute gauge
    gauge = ComputeGauge()
    prev_state = {}
    if state_path:
        try:
            p = Path(state_path)
            if p.exists():
                prev_state = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}
    prev_gauge = prev_state.get("compute_gauge") or {}
    baseline = prev_gauge.get("ops_per_sec") or prev_gauge.get("baseline_ops_per_sec")
    gauge.update_self(fit_stats.get("samples", 0), fit_stats.get("ops", 0), fit_stats.get("elapsed_ns", 1), baseline_ops_per_sec=baseline)

    # self-improving notes (recursive refinement when foreseeable)
    prev_notes = prev_state.get("self_improving_notes", {}).get("notes") if isinstance(prev_state.get("self_improving_notes"), dict) else prev_state.get("notes")
    notes = self_improving_notes(gauge.to_dict(), net_gain, previous_notes=prev_notes, goal=goal)

    result = {
        "neuron": {"weights": neuron.weights, "bias": neuron.bias, "dim": len(neuron.weights)},
        "fit": fit_stats,
        "compressed_net": compressed,
        "log_samples": {
            "weights": weight_log_samples[:16],
            "targets": target_log_samples[:16],
            "base": base,
        },
        "compression": {
            "weight_blob_len": compressed["weights"]["blob_len"],
            "weight_orig_len_est": compressed["weights"]["orig_len_est"],
            "compression_ratio": compression_ratio,
            "reconstructed_weights_preview": recon[:8],
        },
        "compute_gauge": gauge.to_dict(),
        "net_gain": net_gain,
        "self_improving_notes": notes,
    }

    if state_path:
        try:
            p = Path(state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "compute_gauge": gauge.to_dict(),
                "self_improving_notes": notes,
                "net_gain": net_gain,
                "last_input_count": len(qbits),
                "goal": goal,
            }, indent=2), encoding="utf-8")
            result["state_path"] = str(p)
        except Exception as e:
            result["state_error"] = str(e)

    return result
