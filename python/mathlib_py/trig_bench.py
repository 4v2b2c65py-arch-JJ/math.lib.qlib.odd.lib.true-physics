"""Trigonometry benchmark over a telemetry dataset.

Loads data/telemetry_trig_dataset.json, recomputes sin/cos/tan for each sample's
angle, benchmarks throughput/latency, and compares against recorded sensor channels.
Stores results to data/telemetry_trig_bench_result.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List


def _default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "telemetry_trig_dataset.json"


def load_telemetry(path: Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path else _default_dataset_path()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run_trig_bench(dataset: Dict[str, Any]) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = dataset.get("samples", [])
    n = len(samples)

    # Warmup (small) to reduce one-time effects
    for s in samples[: min(32, n)]:
        r = float(s.get("angle_rad", 0.0))
        _ = math.sin(r) + math.cos(r)

    errors_sin: List[float] = []
    errors_cos: List[float] = []
    tan_abs_vals: List[float] = []
    ops = 0

    t0 = time.perf_counter_ns()
    for s in samples:
        rad = float(s.get("angle_rad", 0.0))
        sv = math.sin(rad)
        cv = math.cos(rad)
        # tan guarded like dataset generation
        tv = math.tan(rad) if abs(cv) > 1e-12 else 0.0
        ops += 3

        # compare against sensor channels (dataset stored scaled/noisy values)
        sa = float(s.get("sensor_a", 0.0))
        sb = float(s.get("sensor_b", 0.0))
        # sensor_a ~ sin*100 (+noise), sensor_b ~ cos*50 (+noise)
        if "sensor_a" in s:
            errors_sin.append(abs(sv * 100.0 - sa))
        if "sensor_b" in s:
            errors_cos.append(abs(cv * 50.0 - sb))
        tan_abs_vals.append(abs(tv))
    t1 = time.perf_counter_ns()

    total_ns = int(t1 - t0)
    total_s = total_ns / 1e9 if total_ns > 0 else 0.0
    samples_per_s = (n / total_s) if total_s > 0 else 0.0
    ops_per_s = (ops / total_s) if total_s > 0 else 0.0
    ns_per_sample = (total_ns / n) if n > 0 else 0.0
    ns_per_op = (total_ns / ops) if ops > 0 else 0.0

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "count": len(arr),
            "mean": sum(arr) / len(arr),
            "max": max(arr),
            "min": min(arr),
        }

    result = {
        "benchmark": "trigonometry_telemetry",
        "generated_at_ns": time.time_ns(),
        "dataset": {
            "name": dataset.get("name"),
            "count": dataset.get("count", n),
            "fields": dataset.get("fields", []),
        },
        "workload": {"samples": n, "ops": ops, "ops_per_sample": 3, "functions": ["sin", "cos", "tan"]},
        "timing": {
            "total_ns": total_ns,
            "total_s": total_s,
            "ns_per_sample": ns_per_sample,
            "ns_per_op": ns_per_op,
            "samples_per_s": samples_per_s,
            "ops_per_s": ops_per_s,
        },
        "accuracy_vs_sensors": {
            "sensor_a_sin_scaled_err": stats(errors_sin),
            "sensor_b_cos_scaled_err": stats(errors_cos),
            "note": "Sensors include simulated Gaussian noise; error reflects noise + numeric delta.",
        },
        "tan_magnitude": stats(tan_abs_vals),
    }
    return result


def run_and_store(dataset_path: Path | None = None, out_path: Path | None = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    data = load_telemetry(dataset_path)
    res = run_trig_bench(data)
    out = Path(out_path) if out_path else (root / "data" / "telemetry_trig_bench_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    res["stored_path"] = str(out)
    return res


def main() -> int:
    res = run_and_store()
    summary = {
        "stored_path": res.get("stored_path"),
        "samples": res["workload"]["samples"],
        "total_ns": res["timing"]["total_ns"],
        "samples_per_s": round(res["timing"]["samples_per_s"], 3),
        "ops_per_s": round(res["timing"]["ops_per_s"], 3),
        "sin_err_mean": round(res["accuracy_vs_sensors"]["sensor_a_sin_scaled_err"]["mean"], 6),
        "cos_err_mean": round(res["accuracy_vs_sensors"]["sensor_b_cos_scaled_err"]["mean"], 6),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
