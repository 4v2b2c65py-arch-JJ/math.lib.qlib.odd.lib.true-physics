"""Correct physics constants dataset using SciPy CODATA (installed) and produce
a corrected/normalized dataset for timeline/mathlib use.

Corrections performed:
- load authoritative values from scipy.constants (CODATA 2022 in SciPy 1.14.x)
- normalize selected constants to SI base values/units where applicable
- detect inconsistencies (e.g. derived c from mu0*epsilon0, hbar relation)
- correct/override mismatched derived entries in the exported dataset view
- emit corrected dataset JSON and a correction report
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import scipy
    from scipy import constants as C
except Exception as e:  # pragma: no cover
    raise SystemExit(f"scipy is required for CODATA dataset correction: {e}")


@dataclass
class ConstantRecord:
    key: str
    value: float
    unit: str
    uncertainty: float
    source: str
    corrected: bool = False
    note: str = ""


def _pc(key: str) -> Tuple[float, str, float]:
    v, u, unc = C.physical_constants[key]
    return float(v), str(u), float(unc)


def load_selected_codata() -> Dict[str, ConstantRecord]:
    keys = [
        "speed of light in vacuum",
        "Planck constant",
        "reduced Planck constant",
        "elementary charge",
        "electron mass",
        "proton mass",
        "neutron mass",
        "Boltzmann constant",
        "Avogadro constant",
        "Newtonian constant of gravitation",
        "vacuum electric permittivity",
        "vacuum mag. permeability",
        "fine-structure constant",
        "Rydberg constant",
        "atomic mass constant",
        "standard acceleration of gravity",
    ]
    out: Dict[str, ConstantRecord] = {}
    for k in keys:
        try:
            v, u, unc = _pc(k)
        except KeyError:
            continue
        out[k] = ConstantRecord(k, v, u, unc, source=f"scipy.constants/CODATA2022 (scipy {scipy.__version__})")
    # also include attribute-based canonical ones
    canon = {
        "c_attr": (float(C.c), "m s^-1", 0.0),
        "h_attr": (float(C.h), "J s", 0.0),
        "hbar_attr": (float(C.hbar), "J s", 0.0),
        "e_attr": (float(C.e), "C", 0.0),
        "k_attr": (float(C.k), "J K^-1", 0.0),
        "N_A_attr": (float(C.N_A), "mol^-1", 0.0),
        "G_attr": (float(C.G), "m^3 kg^-1 s^-2", 0.0),
        "epsilon_0_attr": (float(C.epsilon_0), "F m^-1", 0.0),
        "mu_0_attr": (float(C.mu_0), "N A^-2", 0.0),
        "alpha_attr": (float(C.alpha), "", 0.0),
    }
    for k, (v, u, unc) in canon.items():
        out[k] = ConstantRecord(k, v, u, unc, source=f"scipy.constants attribute (scipy {scipy.__version__})")
    return out


def correct_dataset(data: Dict[str, ConstantRecord]) -> Tuple[Dict[str, ConstantRecord], List[dict]]:
    report: List[dict] = []
    c = data.get("speed of light in vacuum") or data.get("c_attr")
    h = data.get("Planck constant") or data.get("h_attr")
    hbar = data.get("reduced Planck constant") or data.get("hbar_attr")
    mu0 = data.get("vacuum mag. permeability") or data.get("mu_0_attr")
    eps0 = data.get("vacuum electric permittivity") or data.get("epsilon_0_attr")

    # 1) hbar consistency: h/(2*pi)
    if h and hbar:
        expected = h.value / (2.0 * math.pi)
        rel = abs(expected - hbar.value) / abs(expected) if expected else 0.0
        ok = rel < 1e-15
        report.append({"check": "hbar_vs_h_over_2pi", "ok": ok, "expected": expected, "actual": hbar.value, "rel_err": rel})
        if not ok:
            hbar.value = expected
            hbar.corrected = True
            hbar.note = (hbar.note + "; corrected from h/(2*pi)").strip("; ")

    # 2) c consistency with mu0*eps0 (SI 2019+ defines c exact but eps0/mu0 derived; still check relation)
    if c and mu0 and eps0:
        try:
            c_from_em = 1.0 / math.sqrt(mu0.value * eps0.value)
            rel = abs(c_from_em - c.value) / c.value
            ok = rel < 1e-12
            report.append({"check": "c_vs_1_sqrt_mu0_eps0", "ok": ok, "expected": c_from_em, "actual": c.value, "rel_err": rel})
        except Exception as e:
            report.append({"check": "c_vs_1_sqrt_mu0_eps0", "ok": False, "error": str(e)})

    # 3) fine-structure alpha consistency (using CODATA values): alpha = e^2/(2*eps0*h*c)
    e_rec = data.get("elementary charge") or data.get("e_attr")
    alpha_rec = data.get("fine-structure constant") or data.get("alpha_attr")
    if e_rec and eps0 and h and c and alpha_rec:
        expected_alpha = (e_rec.value ** 2) / (2.0 * eps0.value * h.value * c.value)
        rel = abs(expected_alpha - alpha_rec.value) / alpha_rec.value if alpha_rec.value else 0.0
        ok = rel < 1e-10
        report.append({"check": "alpha_vs_derived", "ok": ok, "expected": expected_alpha, "actual": alpha_rec.value, "rel_err": rel})

    # 4) normalize: ensure positive uncertainties non-negative and units trimmed
    for rec in data.values():
        if rec.uncertainty < 0:
            rec.uncertainty = abs(rec.uncertainty)
            rec.corrected = True
            rec.note = (rec.note + "; uncertainty sign corrected").strip("; ")
        nu = rec.unit.strip()
        if nu != rec.unit:
            rec.unit = nu
            rec.corrected = True
            rec.note = (rec.note + "; unit trimmed").strip("; ")

    # 5) add derived corrected entries used by timeline/mathlib (SI normalized)
    def add_derived(key, value, unit, note):
        data[key] = ConstantRecord(key, float(value), unit, 0.0, source="derived/corrected from scipy CODATA", corrected=True, note=note)
    if h:
        add_derived("derived.h_eV_s", h.value / C.eV, "eV s", "Planck constant in eV*s")
    if hbar:
        add_derived("derived.hbar_eV_s", hbar.value / C.eV, "eV s", "reduced Planck constant in eV*s")
    if c and h:
        add_derived("derived.hc_eV_m", (h.value * c.value) / C.eV, "eV m", "hc in eV*m")
    report.append({"check": "dataset_normalized", "ok": True, "records": len(data)})
    return data, report


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_selected_codata()
    corrected, report = correct_dataset(data)
    payload = {
        "generated_at_ns": time.time_ns(),
        "scipy_version": scipy.__version__,
        "source": "scipy.constants (CODATA 2022 recommended values as provided by installed SciPy)",
        "install_note": "scipy already installed; CODATA dataset accessed via scipy.constants.physical_constants",
        "records": {k: asdict(v) for k, v in corrected.items()},
        "corrections_report": report,
    }
    out_json = output_dir / "physics_codata_corrected.json"
    out_report = output_dir / "physics_correction_report.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_report.write_text(json.dumps({"generated_at_ns": payload["generated_at_ns"], "report": report}, indent=2), encoding="utf-8")
    return {"dataset": str(out_json), "report_file": str(out_report), "record_count": len(corrected), "checks": report}


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    res = run(here / "data")
    print(json.dumps(res, indent=2, default=str))
