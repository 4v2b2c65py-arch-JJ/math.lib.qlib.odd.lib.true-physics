"""GPU backend connections: Vulkan and CUDA detection/bridge for mathlib.

Probes available runtimes/bindings and provides a unified connection/status API.
Vulkan is detected via Python bindings/env/tools; CUDA is probed in current env
and known environments (e.g. Leviathan directory with CUDA image/venv).
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BackendStatus:
    name: str
    available: bool
    connected: bool
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def _run_cmd(cmd: List[str], timeout: float = 4.0) -> Dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "code": p.returncode, "out": (p.stdout or "")[:4000], "err": (p.stderr or "")[:2000]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def probe_vulkan(deep: bool = False) -> BackendStatus:
    st = BackendStatus(name="vulkan", available=False, connected=False)
    env = {k: os.environ.get(k) for k in ("VULKAN_SDK", "VK_SDK_PATH", "VK_ICD_FILENAMES", "VK_LAYER_PATH") if os.environ.get(k)}
    if env:
        st.details["env"] = env
    vi_path = shutil.which("vulkaninfo")
    st.details["vulkaninfo_path"] = vi_path

    # Python bindings presence (fast, no import side-effects)
    try:
        spec = importlib.util.find_spec("vulkan")
        if spec is not None:
            st.available = True
            st.connected = True
            st.details["python_binding"] = {"module": "vulkan", "origin": getattr(spec, "origin", None)}
    except Exception as e:
        st.errors.append(f"vulkan spec: {e}")

    # Environment/tooling presence implies connectable runtime
    if env or vi_path:
        st.available = True
        st.connected = True

    # Optional deep probe (bounded) + complementary device/card detection
    if deep and vi_path:
        vi = _run_cmd([vi_path, "--summary"], timeout=8.0)
        if vi.get("ok"):
            keys = []
            devices = []
            cur: Dict[str, Any] = {}
            for line in (vi.get("out") or "").splitlines():
                l = line.strip()
                if not l:
                    if cur:
                        devices.append(cur)
                        cur = {}
                    continue
                if any(k in l for k in ("Vulkan Instance Version", "deviceName", "driverName", "driverInfo", "apiVersion", "driverVersion", "deviceType", "deviceID", "vendorID")):
                    keys.append(l)
                if "deviceName" in l and "=" in l:
                    if cur:
                        devices.append(cur)
                    cur = {"deviceName": l.split("=", 1)[1].strip()}
                elif "=" in l and cur:
                    k, v = l.split("=", 1)
                    k = k.strip()
                    if k in ("deviceID", "vendorID", "driverName", "driverInfo", "apiVersion", "driverVersion", "deviceType"):
                        cur[k] = v.strip()
            if cur:
                devices.append(cur)
            if keys:
                st.details["vulkaninfo_summary"] = keys[:40]
            if devices:
                st.details["detected_devices"] = devices[:20]

    if not st.available:
        st.errors.append("Vulkan runtime/bindings not detected.")
    return st


def _find_leviathan_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("LEVIATHAN_PATH") or os.environ.get("LEVIATHAN_DIR")
    if env:
        roots.append(Path(env))
    for c in [Path("F:/leviathan"), Path("F:\\leviathan"), Path.cwd() / "leviathan"]:
        if c not in roots:
            roots.append(c)
    out = []
    seen = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        k = str(rp).lower()
        if k in seen:
            continue
        seen.add(k)
        if rp.exists() and rp.is_dir():
            out.append(rp)
    return out


def _probe_leviathan_cuda(deep: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {"roots": [], "cuda_images": [], "venv_pythons": [], "torch_probes": [], "available": False, "connected": False}
    roots = _find_leviathan_roots()
    if not roots:
        return info
    for root in roots:
        info["roots"].append(str(root))
        # look for CUDA image/binary artifacts (bounded)
        patterns = ["**/*.ptx", "**/*.cubin", "**/*.fatbin", "**/*cuda*.img", "**/*cuda*.image", "**/*cuda*image*", "**/*cuda*.bin"]
        imgs: List[str] = []
        for pat in patterns:
            try:
                for p in root.glob(pat):
                    if p.is_file():
                        imgs.append(str(p))
                        if len(imgs) >= 20:
                            break
            except Exception:
                pass
            if len(imgs) >= 20:
                break
        if imgs:
            info["cuda_images"].extend(imgs)

        # known venv python (leviathan CUDA env)
        for cand in [root / "games" / ".venv" / "Scripts" / "python.exe", root / "games" / ".venv" / "bin" / "python"]:
            if cand.exists():
                s = str(cand)
                if s not in info["venv_pythons"]:
                    info["venv_pythons"].append(s)

    # Leviathan CUDA environment: directory + CUDA venv/image indicates valid CUDA image (works from leviathan)
    if info["roots"] and (info["venv_pythons"] or info["cuda_images"]):
        info["available"] = True
        info["connected"] = True
        info["note"] = "Leviathan CUDA environment/image detected (CUDA available from leviathan directory)."
    elif info["roots"]:
        info["leviathan_detected"] = True

    # optional deep torch probe (bounded, best-effort)
    if deep and info["venv_pythons"]:
        py = info["venv_pythons"][0]
        probe = _run_cmd([py, "-c", "import json,torch;print(json.dumps({'v':torch.__version__,'cuda':bool(torch.cuda.is_available()),'cv':getattr(torch.version,'cuda',None)}))"], timeout=20.0)
        if probe.get("ok"):
            out = (probe.get("out") or "").strip().splitlines()
            for line in reversed(out):
                if line.strip().startswith("{"):
                    try:
                        info["torch_probes"].append({"python": py, "torch": json.loads(line.strip())})
                        info["available"] = True
                        info["connected"] = True
                    except Exception:
                        pass
                    break

    return info


def _find_nvidia_download_roots() -> List[Path]:
    roots: List[Path] = []
    env = os.environ.get("NVIDIA_DOWNLOADS_PATH") or os.environ.get("NVIDIA_PATH")
    if env:
        roots.append(Path(env))
    for c in [Path("E:/DOWNLOADs/NVIDIA"), Path("E:\\DOWNLOADs\\NVIDIA"), Path("E:/Downloads/NVIDIA")]:
        if c not in roots:
            roots.append(c)
    out = []
    seen = set()
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        k = str(rp).lower()
        if k in seen:
            continue
        seen.add(k)
        if rp.exists() and rp.is_dir():
            out.append(rp)
    return out


def _probe_nvidia_downloads_cuda() -> Dict[str, Any]:
    """Probe E:\\DOWNLOADs\\NVIDIA (optimas/driver_extracted) for further CUDA features and detected card IDs."""
    info: Dict[str, Any] = {"roots": [], "features": [], "available": False, "connected": False, "detected_cards": []}
    roots = _find_nvidia_download_roots()
    if not roots:
        return info
    for root in roots:
        info["roots"].append(str(root))
        optimas = root / "optimas"
        disp = optimas / "driver_extracted" / "Display.Driver"
        checks = [
            ("check_cuda.py", optimas / "check_cuda.py"),
            ("nvcuda64", disp / "nvcuda64.dl_"),
            ("nvcuda32", disp / "nvcuda32.dl_"),
            ("nvcuda_loader64", disp / "nvcuda_loader64.dl_"),
            ("nvcuda_loader32", disp / "nvcuda_loader32.dl_"),
            ("libcuda.so.1.1", disp / "libcuda.so.1.1"),
            ("libcuda_loader.so", disp / "libcuda_loader.so"),
            ("nvcuvid64", disp / "nvcuvid64.dl_"),
            ("nvencodeapi64", disp / "nvencodeapi64.dl_"),
            ("nvptxjitcompiler64", disp / "nvptxjitcompiler64.dl_"),
            ("nvml.dll", disp / "nvml.dll"),
            ("nvvm64_40_0.dll", disp / "nvvm64_40_0.dll"),
        ]
        for name, path in checks:
            try:
                if path.exists():
                    info["features"].append({"name": name, "path": str(path)})
            except Exception:
                pass
        for name in ["nvidia_driver_tool.py", "session_driver.py", "driver_install_debug.ps1", "gpu_translator.py", "vulkan_linker.py", "pipeline_linker.py"]:
            p = optimas / name
            try:
                if p.exists() and not any(f.get("path") == str(p) for f in info["features"]):
                    info["features"].append({"name": name, "path": str(p), "kind": "tool"})
            except Exception:
                pass
        # complementary: detect/use cards from optimas live capability/ledger state
        for cname in ("gpu_caps.json", "gpu_ledger.json"):
            cpath = optimas / cname
            try:
                if not cpath.exists():
                    continue
                with cpath.open("r", encoding="utf-8") as f:
                    cdata = json.load(f)
                if cname == "gpu_caps.json" and isinstance(cdata, dict):
                    info["gpu_caps"] = str(cpath)
                    gpus = cdata.get("gpus") or {}
                    for vendor, g in gpus.items():
                        if isinstance(g, dict) and g.get("present"):
                            info["detected_cards"].append({
                                "vendor": vendor,
                                "name": g.get("name"),
                                "device_id": g.get("device_id"),
                                "vendor_id": g.get("vendor_id"),
                                "subsys_id": g.get("subsys_id"),
                                "vram_mb": g.get("vram_mb"),
                                "feature_level": g.get("feature_level"),
                                "caps": g.get("caps"),
                                "source": str(cpath),
                            })
                    if cdata.get("translation"):
                        info["routing_table"] = cdata["translation"].get("routing_table")
                elif cname == "gpu_ledger.json":
                    info["gpu_ledger"] = str(cpath)
            except Exception as e:
                info.setdefault("errors", []).append(f"{cname}: {str(e)[:180]}")
    if info["features"] or info["detected_cards"]:
        info["available"] = True
        info["connected"] = True
        info["note"] = "NVIDIA downloads/optimas: CUDA driver/runtime features and detected card capabilities (complementary)."
    elif info["roots"]:
        info["note"] = "NVIDIA downloads root detected."
    return info


def _probe_nm_resistor() -> Dict[str, Any]:
    """Probe F:\\nm=resistor for extra Vulkan/CUDA compute features."""
    info: Dict[str, Any] = {"root": None, "features": [], "available": False, "connected": False}
    roots = []
    if os.environ.get("NM_RESISTOR_PATH"):
        roots.append(Path(os.environ["NM_RESISTOR_PATH"]))
    roots += [Path("F:/nm=resistor"), Path("F:\\nm=resistor")]
    root = None
    for r in roots:
        try:
            if r.exists() and r.is_dir():
                root = r
                break
        except Exception:
            pass
    if root is None:
        return info
    info["root"] = str(root)
    checks = [
        ("resistor_dispatch", root / "resistor_dispatch.py"),
        ("vulkan_pipeline", root / "vulkan_pipeline.json"),
        ("vulkan_permissions", root / "vulkan_permissions.json"),
        ("mcp_vulkan_permissions", root / "mcp_vulkan_permissions.py"),
        ("fluid_3d_shader", root / "shaders" / "fluid_3d_benchmark.comp"),
    ]
    for name, path in checks:
        try:
            if path.exists():
                info["features"].append({"name": name, "path": str(path)})
        except Exception:
            pass
    caps = []
    if (root / "resistor_dispatch.py").exists():
        caps.extend(["cpu_gpu_dispatch", "cupy_batch", "vulkan_director", "rx580_vulkan_trigger", "auto_batch_handler"])
    if (root / "mcp_vulkan_permissions.py").exists():
        caps.append("vulkan_permission_mcp")
    if (root / "vulkan_pipeline.json").exists():
        caps.append("vulkan_compute_pipeline")
    if caps:
        info["capabilities"] = caps
    if info["features"]:
        info["available"] = True
        info["connected"] = True
        info["note"] = "nm=resistor extra Vulkan/CUDA dispatch features detected."
    return info


def _normalize_gotlet_device(d: Dict[str, Any], source: str) -> Dict[str, Any]:
    out = dict(d)
    out["source"] = source
    out.setdefault("card_id", d.get("uuid") or d.get("serial") or d.get("pci_bus_id") or d.get("device_id") or d.get("name"))
    if "vram_gb" not in out and "vram_mb" in out:
        try:
            out["vram_gb"] = round(float(out["vram_mb"]) / 1024.0, 3)
        except Exception:
            pass
    return out


def _load_nvml_sim_module(path: Path, mod_name: str) -> Dict[str, Any]:
    res: Dict[str, Any] = {"path": str(path), "devices": [], "specs": {}, "errors": []}
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "SERIES_SPECS"):
                res["specs"]["series_specs"] = getattr(mod, "SERIES_SPECS")
            if hasattr(mod, "build_simulator"):
                sim = mod.build_simulator()
                if hasattr(sim, "to_dict"):
                    d = sim.to_dict()
                    res["devices"] = d.get("devices", [])
                    res["cuda_connectors"] = d.get("cuda_connectors", [])
                    res["device_count"] = d.get("device_count", len(res["devices"]))
            elif hasattr(mod, "DEVICES"):
                devs = getattr(mod, "DEVICES")
                if isinstance(devs, list):
                    res["devices"] = devs
    except Exception as e:
        res["errors"].append(str(e)[:300])
    return res


def _load_card_spec_json(path: Path) -> Dict[str, Any]:
    res: Dict[str, Any] = {"path": str(path), "devices": [], "specs": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if isinstance(data.get("devices"), list):
                res["devices"] = data["devices"]
            if isinstance(data.get("series_specs"), dict):
                res["specs"]["series_specs"] = data["series_specs"]
            elif isinstance(data.get("SERIES_SPECS"), dict):
                res["specs"]["series_specs"] = data["SERIES_SPECS"]
            elif any(k in data for k in ("vram_gb", "memory_bus_bits", "cuda_cores", "arch", "compute_cap")):
                res["specs"]["series_specs"] = data
            if not res["devices"] and any(k in data for k in ("name", "serial", "uuid", "pci_bus_id", "device_id", "vram_gb")):
                res["devices"] = [data]
        elif isinstance(data, list):
            res["devices"] = data
    except Exception as e:
        res["error"] = str(e)[:300]
    return res


def _probe_gotlet_card_hub() -> Dict[str, Any]:
    """Probe Gotlet (card hub) for card ID/VRAM/spec resolution, including complementary/new cards."""
    info: Dict[str, Any] = {"root": None, "available": False, "connected": False, "devices": [], "specs": {}, "sources": [], "complementary": []}
    roots = []
    custom = os.environ.get("GOTLET_NVML_SIM_PATH")
    if os.environ.get("GOTLET_PATH"):
        roots.append(Path(os.environ["GOTLET_PATH"]))
    roots += [Path("F:/Gotlet"), Path("F:\\Gotlet"), Path("F:/Gotlet card hub"), Path("F:\\Gotlet card hub")]
    root = None
    for r in roots:
        try:
            if r.exists() and r.is_dir():
                root = r
                break
        except Exception:
            pass
    if root is None and custom:
        # custom module path may exist even without a standard root
        cp = Path(custom)
        if cp.exists():
            info["custom_nvml_sim"] = str(cp)
            loaded = _load_nvml_sim_module(cp, "gotlet_nvml_custom")
            info["sources"].append(loaded.get("path"))
            for d in loaded.get("devices", []):
                info["devices"].append(_normalize_gotlet_device(d, str(cp)))
            for k, v in (loaded.get("specs") or {}).items():
                info["specs"].setdefault(k, {}).update(v if isinstance(v, dict) else {})
            if loaded.get("errors"):
                info.setdefault("errors", []).extend(loaded["errors"])
    if root is None:
        return info
    info["root"] = str(root)
    info["files"] = [p.name for p in [root / "telemetry.py", root / "telemetry.json", root / "bench_api.py"] if p.exists()]

    sim_paths: List[Path] = []
    if custom:
        sim_paths.append(Path(custom))
    primary = root / "activity" / "QBM-" / "_nvml_sim.py"
    if primary.exists():
        sim_paths.append(primary)
    try:
        for p in sorted((root / "activity").glob("**/*nvml*sim*.py")) if (root / "activity").exists() else []:
            if p not in sim_paths:
                sim_paths.append(p)
        for p in sorted(root.glob("*nvml*sim*.py")):
            if p not in sim_paths:
                sim_paths.append(p)
    except Exception:
        pass

    seen_paths = set()
    for i, sp in enumerate(sim_paths):
        try:
            if not sp.exists() or not sp.is_file():
                continue
            key = str(sp.resolve()).lower()
        except Exception:
            key = str(sp).lower()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        loaded = _load_nvml_sim_module(sp, f"gotlet_nvml_sim_{i}")
        info["sources"].append(str(sp))
        for d in loaded.get("devices", []):
            info["devices"].append(_normalize_gotlet_device(d, str(sp)))
        for k, v in (loaded.get("specs") or {}).items():
            if isinstance(v, dict):
                info["specs"].setdefault(k, {}).update(v)
        if loaded.get("cuda_connectors"):
            info.setdefault("cuda_connectors", [])
            for c in loaded["cuda_connectors"]:
                if c not in info["cuda_connectors"]:
                    info["cuda_connectors"].append(c)
        if loaded.get("errors"):
            info.setdefault("errors", []).extend(loaded["errors"])
        if str(sp) == str(primary):
            info["nvml_sim"] = str(sp)

    # complementary card/spec JSONs (custom or new cards)
    json_candidates: List[Path] = []
    for envk in ("GOTLET_CARD_SPECS_PATH", "GOTLET_CARDS_JSON"):
        ev = os.environ.get(envk)
        if ev:
            json_candidates.append(Path(ev))
    try:
        for pat in ("*card*spec*.json", "*card*.json", "*gpu*spec*.json", "*device*spec*.json"):
            for p in root.glob(pat):
                if p not in json_candidates:
                    json_candidates.append(p)
            act = root / "activity"
            if act.exists():
                for p in act.glob(pat):
                    if p not in json_candidates:
                        json_candidates.append(p)
    except Exception:
        pass
    seen_json = set()
    for jp in json_candidates:
        try:
            if not jp.exists() or not jp.is_file():
                continue
            jk = str(jp.resolve()).lower()
        except Exception:
            jk = str(jp).lower()
        if jk in seen_json:
            continue
        seen_json.add(jk)
        loaded_j = _load_card_spec_json(jp)
        if loaded_j.get("error"):
            continue
        info["complementary"].append(str(jp))
        for d in loaded_j.get("devices", []):
            info["devices"].append(_normalize_gotlet_device(d, str(jp)))
        for k, v in (loaded_j.get("specs") or {}).items():
            if isinstance(v, dict):
                info["specs"].setdefault(k, {}).update(v)

    # de-duplicate devices by card_id/name+pci
    uniq = []
    seen_d = set()
    for d in info.get("devices", []):
        did = str(d.get("card_id") or d.get("uuid") or d.get("serial") or d.get("pci_bus_id") or d.get("name") or id(d)).lower()
        if did in seen_d:
            continue
        seen_d.add(did)
        uniq.append(d)
    info["devices"] = uniq
    info["device_count"] = len(uniq)

    if info.get("sources") or info.get("complementary") or info.get("files") or info.get("devices") or info.get("specs"):
        info["available"] = True
        info["connected"] = True
        info["note"] = "Gotlet card hub: resolve card ID, VRAM, and specs (including complementary sources) via NVML simulators/spec files/telemetry."
    return info


def probe_cuda(deep: bool = False) -> BackendStatus:
    st = BackendStatus(name="cuda", available=False, connected=False)
    st.details["nvcc_path"] = shutil.which("nvcc")
    st.details["nvidia_smi_path"] = shutil.which("nvidia-smi")

    # fast checks in current env (no heavy imports by default)
    try:
        if importlib.util.find_spec("torch") is not None:
            st.details["torch"] = {"installed": True}
            st.available = True
    except Exception:
        pass
    try:
        if importlib.util.find_spec("cupy") is not None:
            st.details["cupy"] = {"installed": True}
            st.available = True
    except Exception:
        pass

    # Leviathan environment (CUDA image/venv)
    lev = _probe_leviathan_cuda(deep=deep)
    if lev.get("roots") or lev.get("venv_pythons") or lev.get("cuda_images"):
        st.details["leviathan"] = lev
    if lev.get("available") or lev.get("connected"):
        st.available = True
        st.connected = True

    # NVIDIA downloads (further CUDA features, e.g. E:\DOWNLOADs\NVIDIA\optimas)
    nvd = _probe_nvidia_downloads_cuda()
    if nvd.get("roots") or nvd.get("features"):
        st.details["nvidia_downloads"] = nvd
    if nvd.get("available") or nvd.get("connected"):
        st.available = True
        st.connected = True

    # nm=resistor extra Vulkan/CUDA features
    nmr = _probe_nm_resistor()
    if nmr.get("root") or nmr.get("features"):
        st.details["nm_resistor"] = nmr
    if nmr.get("available") or nmr.get("connected"):
        st.available = True
        st.connected = True

    # Gotlet card hub (resolve card ID, VRAM, specs)
    got = _probe_gotlet_card_hub()
    if got.get("root") or got.get("devices") or got.get("specs"):
        st.details["gotlet_card_hub"] = got
    if got.get("available") or got.get("connected"):
        st.available = True
        st.connected = True

    if deep:
        smi = _run_cmd(["nvidia-smi", "-L"], timeout=5.0)
        if smi.get("ok"):
            st.details["nvidia_smi"] = smi.get("out", "").strip().splitlines()[:16]
            st.available = True

    if not st.available:
        st.errors.append("No CUDA-capable runtime/bindings detected in current env, leviathan, NVIDIA downloads, nm=resistor, or Gotlet.")

    return st


def connect_backends(prefer: Optional[str] = None, deep: bool = False) -> Dict[str, Any]:
    """Probe/connect Vulkan and CUDA and return unified status."""
    vk = probe_vulkan(deep=deep)
    cu = probe_cuda(deep=deep)
    available = [b.name for b in (vk, cu) if b.available or b.connected]
    selected = None
    if prefer in ("vulkan", "cuda"):
        cand = vk if prefer == "vulkan" else cu
        selected = cand.name if (cand.available or cand.connected) else None
    if selected is None:
        if cu.connected or cu.available:
            selected = "cuda"
        elif vk.connected or vk.available:
            selected = "vulkan"
    return {
        "generated_at_ns": time.time_ns(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "selected": selected,
        "available_backends": available,
        "vulkan": asdict(vk),
        "cuda": asdict(cu),
        "interop": {
            "vulkan_cuda_interop_possible": bool(vk.connected and cu.connected),
            "note": "Vulkan-CUDA interop (external memory/semaphore) requires compatible NVIDIA driver, CUDA runtime, and Vulkan extensions; not available on all systems.",
        },
    }


def _load_json_arg(val: Optional[str]) -> Any:
    if val is None:
        return None
    v = str(val)
    if v.startswith("@"):
        p = Path(v[1:])
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        return json.loads(v)
    except Exception:
        return val


def _numpy_from_payload(payload: Any):
    import numpy as np  # local import
    if isinstance(payload, str) and payload.startswith("@"):
        return _numpy_from_payload(_load_json_arg(payload))
    if isinstance(payload, dict):
        if "file" in payload:
            p = Path(str(payload["file"]))
            suf = p.suffix.lower()
            if suf == ".npy":
                arr = np.load(str(p), allow_pickle=False)
            elif suf == ".npz":
                data = np.load(str(p), allow_pickle=False)
                key = payload.get("key") or (data.files[0] if getattr(data, "files", None) else None)
                arr = data[key] if key is not None else np.array([])
            elif suf == ".json":
                arr = np.array(json.loads(p.read_text(encoding="utf-8")))
            else:
                # raw bytes, requires dtype/shape
                dt = np.dtype(payload.get("dtype", "uint8"))
                raw = p.read_bytes()
                arr = np.frombuffer(raw, dtype=dt)
                if "shape" in payload:
                    arr = arr.reshape(tuple(int(x) for x in payload["shape"]))
            return np.ascontiguousarray(arr)
        if "data" in payload:
            arr = np.array(payload["data"], dtype=payload.get("dtype")) if "dtype" in payload else np.array(payload["data"])
            if "shape" in payload:
                arr = arr.reshape(tuple(int(x) for x in payload["shape"]))
            return np.ascontiguousarray(arr)
    if isinstance(payload, (list, tuple)):
        return np.ascontiguousarray(np.array(payload))
    return np.ascontiguousarray(np.array(payload))


def _save_staged(arr, out: Optional[str]) -> Dict[str, Any]:
    import numpy as np  # noqa
    meta = {"shape": list(arr.shape), "dtype": str(arr.dtype), "nbytes": int(arr.nbytes)}
    if not out:
        return {"staged": False, **meta}
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf == ".npy":
        import numpy as np
        np.save(str(p), arr, allow_pickle=False)
    elif suf == ".npz":
        import numpy as np
        np.savez(str(p), data=arr)
    elif suf == ".json":
        p.write_text(json.dumps(arr.tolist()), encoding="utf-8")
    else:
        # default raw binary
        p.write_bytes(np.ascontiguousarray(arr).tobytes())
    meta.update({"staged": True, "path": str(p)})
    return meta


def _run_in_python(python_exe: str, code: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    try:
        p = subprocess.run([python_exe, "-c", code], input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return {"ok": False, "code": p.returncode, "err": (p.stderr or "")[-2000:], "out": (p.stdout or "")[-2000:]}
        out = (p.stdout or "").strip().splitlines()
        for line in reversed(out):
            if line.strip().startswith("{"):
                try:
                    return {"ok": True, "result": json.loads(line.strip())}
                except Exception:
                    pass
        return {"ok": False, "err": "no-json-output", "out": (p.stdout or "")[-2000:]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def transfer_vram(data: Any = None, source: str = "auto", target: str = "host", out: Optional[str] = None, device: Optional[str] = None, deep: bool = False) -> Dict[str, Any]:
    """Host-staged VRAM/buffer assistance transfer (explicit, no direct cross-vendor VRAM offload).

    Stages data through host memory and can execute CUDA-side staging in Leviathan torch env when requested.
    Direct RX580/AMD VRAM -> NVIDIA CUDA VRAM offload is not supported; use source=vulkan/host staging + target=cuda.
    """
    import numpy as np
    status = connect_backends(deep=deep)
    res: Dict[str, Any] = {
        "ok": False,
        "generated_at_ns": time.time_ns(),
        "source": source,
        "target": target,
        "host_staged": True,
        "backends": status,
        "steps": [],
        "warnings": [],
    }
    try:
        arr = _numpy_from_payload(data) if data is not None else None
    except Exception as e:
        res["error"] = f"payload_load_failed: {e}"
        return res

    src = (source or "auto").lower()
    tgt = (target or "host").lower()

    # Vulkan source: no direct GPU readback path implemented; use host-staged payload/export from Vulkan compute path (e.g. nm=resistor).
    if src in ("vulkan", "rx580", "amd"):
        res["steps"].append({"stage": "vulkan_readback", "mode": "host-staged", "status": "contract-only"})
        nmr = (status.get("cuda", {}).get("details", {}) or {}).get("nm_resistor") or {}
        if nmr:
            res["nm_resistor"] = nmr
        if arr is None:
            res["warnings"].append("No host payload provided for Vulkan->host staging. Export/readback must be produced by the Vulkan compute path (e.g. nm=resistor shader/dispatch) and passed as data/file.")
        else:
            res["steps"].append({"stage": "host_stage", "status": "ok"})

    # Host/auto staging
    if arr is not None:
        stage_meta = _save_staged(arr, out if tgt == "host" else None)
        res["host_buffer"] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "nbytes": int(arr.nbytes)}
        res["steps"].append({"stage": "host_stage", "status": "ok", "meta": stage_meta})
    elif tgt != "host":
        res["warnings"].append("No data payload to transfer.")

    # CUDA target: best-effort staging via Leviathan torch env (real CUDA runtime if available there)
    if tgt == "cuda":
        if src in ("vulkan", "rx580", "amd"):
            res["warnings"].append("Direct Vulkan/RX580 VRAM to CUDA VRAM offload is unsupported; performing host-staged transfer to CUDA environment.")
        lev = (((status.get("cuda") or {}).get("details") or {}).get("leviathan") or {})
        py = None
        vps = lev.get("venv_pythons") or []
        if vps:
            py = vps[0]
        if not py or arr is None:
            res["steps"].append({"stage": "cuda_stage", "status": "skipped", "reason": "no_leviathan_python_or_payload", "leviathan_python": py})
        else:
            code = r"""
import json, sys, numpy as np
p = json.loads(sys.stdin.read())
arr = np.array(p['data'], dtype=p.get('dtype'))
arr = np.ascontiguousarray(arr)
out = {'ok': True, 'host_nbytes': int(arr.nbytes), 'shape': list(arr.shape), 'dtype': str(arr.dtype), 'cuda_available': False}
try:
    import torch
    out['torch'] = getattr(torch, '__version__', None)
    out['torch_cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', None)
    out['cuda_available'] = bool(torch.cuda.is_available())
    if out['cuda_available']:
        dev = p.get('device') or 'cuda:0'
        t = torch.from_numpy(arr.copy())
        t = t.to(dev)
        torch.cuda.synchronize()
        out['cuda_device'] = dev
        out['cuda_tensor_nbytes'] = int(t.element_size() * t.nelement())
        out['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        # roundtrip to host to confirm staged path works
        back = t.to('cpu').numpy()
        out['roundtrip_ok'] = bool(back.shape == arr.shape)
except Exception as e:
    out['cuda_error'] = str(e)
print(json.dumps(out))
"""
            payload = {"data": arr.tolist(), "dtype": str(arr.dtype), "device": device}
            r = _run_in_python(py, code, payload, timeout=120.0)
            res["steps"].append({"stage": "cuda_stage", "python": py, **r})
            if r.get("ok") and (r.get("result") or {}).get("cuda_available"):
                res["ok"] = True
            elif r.get("ok"):
                res["ok"] = True
                res["warnings"].append("CUDA runtime/device not available in Leviathan env; host staging completed only.")
            else:
                res["warnings"].append("CUDA staging attempt failed; host staging may still be available.")
            if out and tgt == "cuda" and arr is not None:
                # also persist host-staged copy for engine contract handoff
                res["host_copy"] = _save_staged(arr, out)

    elif tgt == "host":
        res["ok"] = arr is not None
        if arr is None:
            res["error"] = "no_data"

    elif tgt in ("vulkan", "rx580", "amd"):
        res["warnings"].append("Upload-to-Vulkan path is not implemented; host staging completed for handoff to Vulkan compute executor (nm=resistor).")
        res["ok"] = arr is not None

    res["interop_possible"] = bool((status.get("interop") or {}).get("vulkan_cuda_interop_possible"))
    if src in ("vulkan", "rx580", "amd") and tgt == "cuda":
        res["note"] = "Used host-staged path (readback/export -> host -> CUDA env). Direct cross-vendor VRAM offload is not possible; interop requires compatible NVIDIA Vulkan+CUDA support."
    return res


def _cli_transfer(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mathlib_py gpu-transfer", description="Host-staged GPU/VRAM assistance transfer (probe/report + staged transfer).")
    ap.add_argument("--data", help="Inline JSON array/object or @path to .npy/.npz/.json/raw payload")
    ap.add_argument("--file", help="Path to payload file (.npy/.npz/.json/raw)")
    ap.add_argument("--dtype", help="dtype for raw file payloads")
    ap.add_argument("--shape", help="comma-separated shape for raw file payloads")
    ap.add_argument("--source", default="auto", help="auto|host|vulkan|rx580|cuda")
    ap.add_argument("--target", default="host", help="host|cuda|vulkan")
    ap.add_argument("--device", help="CUDA device, e.g. cuda:0")
    ap.add_argument("--out", help="Output path for staged host buffer/copy")
    ap.add_argument("--deep", action="store_true", help="Deep backend probe")
    ns = ap.parse_args(argv)
    payload = None
    if ns.file:
        payload = {"file": ns.file}
        if ns.dtype:
            payload["dtype"] = ns.dtype
        if ns.shape:
            payload["shape"] = [int(x) for x in ns.shape.split(",") if x.strip()]
    elif ns.data:
        payload = _load_json_arg(ns.data)
    res = transfer_vram(data=payload, source=ns.source, target=ns.target, out=ns.out, device=ns.device, deep=ns.deep)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("ok") else 1


def main() -> int:
    argv = list(sys.argv[1:])
    if argv and argv[0] in ("transfer", "move", "vram", "stage"):
        return _cli_transfer(argv[1:])
    deep = "--deep" in argv or "-d" in argv
    res = connect_backends(deep=deep)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
