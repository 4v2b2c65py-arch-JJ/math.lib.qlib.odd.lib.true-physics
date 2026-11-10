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

    # Optional deep probe (bounded)
    if deep and vi_path:
        vi = _run_cmd([vi_path, "--summary"], timeout=8.0)
        if vi.get("ok"):
            keys = []
            for line in (vi.get("out") or "").splitlines():
                l = line.strip()
                if any(k in l for k in ("Vulkan Instance Version", "deviceName", "driverName", "driverInfo", "apiVersion", "driverVersion", "deviceType")):
                    keys.append(l)
            if keys:
                st.details["vulkaninfo_summary"] = keys[:30]

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
    """Probe E:\\DOWNLOADs\\NVIDIA (optimas/driver_extracted) for further CUDA features."""
    info: Dict[str, Any] = {"roots": [], "features": [], "available": False, "connected": False}
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
        for name in ["nvidia_driver_tool.py", "session_driver.py", "driver_install_debug.ps1"]:
            p = optimas / name
            try:
                if p.exists() and not any(f.get("path") == str(p) for f in info["features"]):
                    info["features"].append({"name": name, "path": str(p), "kind": "tool"})
            except Exception:
                pass
    if info["features"]:
        info["available"] = True
        info["connected"] = True
        info["note"] = "NVIDIA downloads (E:\\DOWNLOADs\\NVIDIA) provide further CUDA driver/runtime features and tools."
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


def _probe_gotlet_card_hub() -> Dict[str, Any]:
    """Probe Gotlet (card hub) for card ID/VRAM/spec resolution."""
    info: Dict[str, Any] = {"root": None, "available": False, "connected": False, "devices": [], "specs": {}}
    roots = []
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
    if root is None:
        return info
    info["root"] = str(root)
    info["files"] = [p.name for p in [root / "telemetry.py", root / "telemetry.json", root / "bench_api.py"] if p.exists()]

    nvml_path = root / "activity" / "QBM-" / "_nvml_sim.py"
    if nvml_path.exists():
        info["nvml_sim"] = str(nvml_path)
        try:
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location("gotlet_nvml_sim", str(nvml_path))
            if spec and spec.loader:
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "SERIES_SPECS"):
                    info["specs"]["series_specs"] = getattr(mod, "SERIES_SPECS")
                if hasattr(mod, "build_simulator"):
                    sim = mod.build_simulator()
                    if hasattr(sim, "to_dict"):
                        d = sim.to_dict()
                        info["devices"] = d.get("devices", [])
                        info["cuda_connectors"] = d.get("cuda_connectors", [])
                        info["device_count"] = d.get("device_count", len(info["devices"]))
        except Exception as e:
            info["nvml_error"] = str(e)[:300]

    if info.get("nvml_sim") or info.get("files") or info.get("devices"):
        info["available"] = True
        info["connected"] = True
        info["note"] = "Gotlet card hub: resolve card ID, VRAM, and specs via NVML simulator/telemetry."
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


def main() -> int:
    res = connect_backends()
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
