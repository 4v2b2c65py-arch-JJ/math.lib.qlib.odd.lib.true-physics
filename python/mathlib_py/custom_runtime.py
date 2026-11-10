"""Custom GPU/runtime configuration and execution for mathlib_py.

Provides a runtime that can be configured at runtime (paths, defaults, devices)
to probe backends, detect complementary cards, and perform host-staged VRAM
assistance transfers using gpu_backends.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeConfig:
    name: str = "custom"
    prefer: Optional[str] = None  # vulkan|cuda|None
    default_source: str = "auto"  # auto|host|vulkan|rx580|cuda
    default_target: str = "host"  # host|cuda|vulkan
    device: Optional[str] = None  # e.g. cuda:0
    deep: bool = False
    leviathan_path: Optional[str] = None
    leviathan_python: Optional[str] = None
    gotlet_path: Optional[str] = None
    gotlet_nvml_sim: Optional[str] = None
    gotlet_card_specs: Optional[str] = None
    nm_resistor_path: Optional[str] = None
    nvidia_downloads_path: Optional[str] = None
    optimas_path: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)
    transfer_defaults: Dict[str, Any] = field(default_factory=dict)
    # Mock/force-load configuration
    enable_mocks: bool = False
    mock_devices: List[Dict[str, Any]] = field(default_factory=list)
    force_load: bool = False
    force_load_functions: List[str] = field(default_factory=list)
    require_null_for_place_turn: bool = True
    place_turn_config: Dict[str, Any] = field(default_factory=dict)
    lumes: Any = "placeholder"  # placeholder

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "RuntimeConfig":
        d = dict(d or {})
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in fields}
        if "source" in d and "default_source" not in kwargs:
            kwargs["default_source"] = d["source"]
        if "target" in d and "default_target" not in kwargs:
            kwargs["default_target"] = d["target"]
        return cls(**kwargs)

    @classmethod
    def from_json(cls, value: Any) -> "RuntimeConfig":
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls.from_dict(value)
        s = str(value)
        if s.startswith("@"):
            p = Path(s[1:])
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        p = Path(s)
        if p.exists() and p.is_file():
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        return cls.from_dict(json.loads(s))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CustomRuntime:
    """Runtime-configurable GPU/backend probe and host-staged transfer executor."""

    ENV_MAP = {
        "leviathan_path": "LEVIATHAN_PATH",
        "gotlet_path": "GOTLET_PATH",
        "gotlet_nvml_sim": "GOTLET_NVML_SIM_PATH",
        "gotlet_card_specs": "GOTLET_CARD_SPECS_PATH",
        "nm_resistor_path": "NM_RESISTOR_PATH",
        "nvidia_downloads_path": "NVIDIA_DOWNLOADS_PATH",
    }

    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.created_at_ns = time.time_ns()
        self._mock_state: Dict[str, Any] = {}
        self._force_loaded: List[str] = []

    @classmethod
    def from_config(cls, value: Any = None) -> "CustomRuntime":
        return cls(RuntimeConfig.from_json(value))

    def apply_env(self) -> Dict[str, Optional[str]]:
        prev: Dict[str, Optional[str]] = {}
        cfg = self.config
        for attr, envk in self.ENV_MAP.items():
            val = getattr(cfg, attr, None)
            if val:
                prev[envk] = os.environ.get(envk)
                os.environ[envk] = str(val)
        if cfg.optimas_path:
            prev["NVIDIA_DOWNLOADS_PATH"] = os.environ.get("NVIDIA_DOWNLOADS_PATH")
            base = str(Path(cfg.optimas_path).parent) if str(cfg.optimas_path).lower().endswith("optimas") else str(cfg.optimas_path)
            os.environ["NVIDIA_DOWNLOADS_PATH"] = base
        for k, v in (cfg.extra_env or {}).items():
            prev[k] = os.environ.get(k)
            os.environ[k] = str(v)
        # Mock/force-load env
        if cfg.enable_mocks:
            os.environ["MATHLIB_MOCK_DEVICES"] = "1"
            if cfg.mock_devices:
                os.environ["MATHLIB_MOCK_DEVICES_JSON"] = json.dumps(cfg.mock_devices)
        if cfg.force_load:
            os.environ["MATHLIB_FORCE_LOAD"] = "1"
            if cfg.force_load_functions:
                os.environ["MATHLIB_FORCE_LOAD_FUNCTIONS"] = json.dumps(cfg.force_load_functions)
        if cfg.require_null_for_place_turn:
            os.environ["MATHLIB_REQUIRE_NULL_FOR_PLACE_TURN"] = "1"
        if cfg.place_turn_config:
            os.environ["MATHLIB_PLACE_TURN_CONFIG"] = json.dumps(cfg.place_turn_config)
        if cfg.lumes is not None:
            os.environ["MATHLIB_LUMES"] = cfg.lumes if isinstance(cfg.lumes, str) else json.dumps(cfg.lumes)
        return prev

    def restore_env(self, prev: Dict[str, Optional[str]]) -> None:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _gb(self):
        from . import gpu_backends
        return gpu_backends

    def probe(self, deep: Optional[bool] = None, prefer: Optional[str] = None) -> Dict[str, Any]:
        prev = self.apply_env()
        try:
            gb = self._gb()
            d = self.config.deep if deep is None else bool(deep)
            p = self.config.prefer if prefer is None else prefer
            status = gb.connect_backends(prefer=p, deep=d)
            status = self._inject_mocks(status)
            status["runtime"] = {"name": self.config.name, "config": self.config.to_dict()}
            if self.config.force_load:
                status = self._force_load_functions(status)
            if self.config.require_null_for_place_turn:
                status = self._configure_place_turn(status)
            return status
        finally:
            self.restore_env(prev)

    def _inject_mocks(self, status: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        if not cfg.enable_mocks:
            return status
        mocks = list(cfg.mock_devices) if cfg.mock_devices else [
            {
                "type": "cuda",
                "name": "Mock NVIDIA Device (forced)",
                "index": 0,
                "uuid": "GPU-MOCK-0000-0000-0000-000000000000",
                "pci_bus_id": "0000:01:00.0",
                "vram_gb": 24,
                "vram_mb": 24576,
                "driver_version": "mock",
                "compute_cap": [9, 0],
            },
            {
                "type": "vulkan",
                "name": "Mock RX 580 (forced load)",
                "vendor_id": "0x1002",
                "device_id": "0x67DF",
                "vram_mb": 8192,
            },
        ]
        status["mocks"] = {"enabled": True, "devices": mocks}
        cuda = status.setdefault("cuda", {"name": "cuda", "available": False, "connected": False, "details": {}, "errors": []})
        cuda_d = cuda.setdefault("details", {})
        cuda_m = [m for m in mocks if str(m.get("type", "")).lower() in ("cuda", "nvidia")]
        if cuda_m:
            cuda_d["mocked_devices"] = cuda_m
            cuda_d["mocked"] = True
            cuda["available"] = True
            cuda["connected"] = True
        vk = status.setdefault("vulkan", {"name": "vulkan", "available": False, "connected": False, "details": {}, "errors": []})
        vk_d = vk.setdefault("details", {})
        vk_m = [m for m in mocks if str(m.get("type", "")).lower() in ("vulkan", "amd", "rx580")]
        if vk_m:
            vk_d["mocked_devices"] = vk_m
            vk_d["mocked"] = True
            vk["available"] = True
            vk["connected"] = True
        if not status.get("selected") and (cuda.get("available") or vk.get("available")):
            status["selected"] = "cuda" if cuda.get("available") else "vulkan"
        status.setdefault("available_backends", [])
        for b in ("cuda", "vulkan"):
            if status.get(b, {}).get("available") and b not in status["available_backends"]:
                status["available_backends"].append(b)
        self._mock_state = {"enabled": True, "devices": mocks}
        # Glow-stick: attach mocks to loader
        self._attach_glow_stick_loader(status, mocks)
        return status

    def _attach_glow_stick_loader(self, status: Dict[str, Any], mocks: Optional[list] = None) -> Dict[str, Any]:
        """Attach ('glow stick') mock devices to a runtime loader."""
        devices = list(mocks if mocks is not None else (self.config.mock_devices or []))
        if not devices and self.config.enable_mocks:
            # ensure defaults available
            devices = (status.get("mocks") or {}).get("devices") or []
        loader = {
            "name": "glow_stick_loader",
            "attached": True,
            "glow_stick": True,
            "mock_count": len(devices),
            "devices": devices,
            "attached_at_ns": time.time_ns(),
            "loader_id": f"glow-loader-{self.created_at_ns}",
            "lumes": self.config.lumes,
        }
        self._mock_state["glow_stick_loader"] = loader
        self._mock_state["glow_sticked"] = True
        status["loader"] = loader
        # expose in details for consumers expecting loader-attached mocks
        for b in ("cuda", "vulkan"):
            if b in status and isinstance(status[b], dict):
                d = status[b].setdefault("details", {})
                if self.config.enable_mocks or devices:
                    d["glow_stick_loader"] = {"attached": True, "mock_count": len(devices)}
        os.environ["MATHLIB_MOCK_GLOW_STICK_LOADER"] = "1"
        return status

    def _force_load_functions(self, status: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        funcs = list(cfg.force_load_functions) if cfg.force_load_functions else [
            "place_turn",
            "force_load",
            "load_device_mock",
            "glow_stick_moc_to_loader",
            "initialize_mock_runtime",
        ]
        loaded = []
        for f in funcs:
            entry: Dict[str, Any] = {"name": f}
            try:
                if f == "place_turn":
                    entry.update({"status": "loaded", "callable": callable(globals().get("place_turn"))})
                elif f == "force_load":
                    entry.update({"status": "loaded", "callable": callable(globals().get("force_load"))})
                elif f == "load_device_mock":
                    self._mock_state.setdefault("force_loaded", True)
                    entry.update({"status": "loaded", "callable": True})
                elif f == "glow_stick_moc_to_loader":
                    if cfg.enable_mocks or self._mock_state.get("devices"):
                        self._attach_glow_stick_loader(status, self._mock_state.get("devices") or cfg.mock_devices)
                    entry.update({"status": "loaded", "callable": True, "attached": bool((status.get("loader") or {}).get("attached"))})
                elif f == "initialize_mock_runtime":
                    self._mock_state.setdefault("initialized", True)
                    entry.update({"status": "loaded", "callable": True})
                else:
                    fn = globals().get(f)
                    entry.update({"status": "loaded" if callable(fn) else "registered", "callable": callable(fn)})
            except Exception as e:
                entry.update({"status": "error", "error": str(e)})
            loaded.append(entry)
        self._force_loaded = [x["name"] for x in loaded]
        status["force_load"] = {"enabled": True, "functions": loaded}
        # ensure glow-stick loader present when mocks enabled
        if cfg.enable_mocks and not status.get("loader"):
            self._attach_glow_stick_loader(status, self._mock_state.get("devices") or cfg.mock_devices)
        return status

    def _configure_place_turn(self, status: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        pconf = dict(cfg.place_turn_config) if cfg.place_turn_config else {}
        pconf.setdefault("require_null", True)
        pconf.setdefault("null_token", None)
        pconf.setdefault("allow_mock_devices", bool(cfg.enable_mocks))
        turns = pconf.get("turns")
        if not isinstance(turns, list):
            turns = []
        pconf["turns"] = turns
        cfg.place_turn_config = pconf
        status["place_turn"] = {"configured": True, "require_null": bool(pconf.get("require_null", True)), "turns": len(turns), "config": pconf}
        os.environ["MATHLIB_PLACE_TURN_REQUIRE_NULL"] = "1" if pconf.get("require_null", True) else "0"
        return status

    def force_load(self, functions: Optional[list] = None) -> Dict[str, Any]:
        """Force-load configured functions (mock/runtime helpers) and return status."""
        if functions is not None:
            self.config.force_load = True
            self.config.force_load_functions = list(functions)
        st = {"runtime": self.config.name, "config": self.config.to_dict()}
        st = self._inject_mocks(st) if self.config.enable_mocks else st
        st = self._force_load_functions(st)
        st = self._configure_place_turn(st) if self.config.require_null_for_place_turn else st
        return {"ok": True, **st}

    def place_turn(self, turn: Any = None, slot: Any = None, **kwargs) -> Dict[str, Any]:
        """Place a turn into runtime state. Requires null slot when configured."""
        require_null = bool(self.config.require_null_for_place_turn)
        if require_null and slot is not None:
            return {
                "ok": False,
                "error": "null_required",
                "message": "place_turn requires null slot (None) to place turns in this runtime configuration",
                "slot": slot,
                "turn": turn,
            }
        pconf = dict(self.config.place_turn_config or {})
        turns = pconf.get("turns")
        if not isinstance(turns, list):
            turns = []
        entry = {
            "turn": turn,
            "slot": None,
            "placed_at_ns": time.time_ns(),
            "meta": kwargs.get("meta"),
        }
        turns.append(entry)
        pconf["turns"] = turns
        self.config.place_turn_config = pconf
        return {"ok": True, "placed": entry, "turn_count": len(turns), "require_null": require_null}

    def detect_cards(self, deep: bool = False) -> Dict[str, Any]:
        st = self.probe(deep=deep)
        cards: Dict[str, Any] = {"runtime": self.config.name, "cuda": [], "vulkan": [], "gotlet": [], "optimas": []}
        cuda_d = (st.get("cuda") or {}).get("details") or {}
        vk_d = (st.get("vulkan") or {}).get("details") or {}
        nvd = cuda_d.get("nvidia_downloads") or {}
        got = cuda_d.get("gotlet_card_hub") or {}
        cards["optimas"] = nvd.get("detected_cards") or []
        cards["gotlet"] = got.get("devices") or []
        cards["gotlet_specs"] = (got.get("specs") or {}).get("series_specs")
        cards["vulkan"] = vk_d.get("detected_devices") or []
        cards["selected"] = st.get("selected")
        cards["routing_table"] = nvd.get("routing_table")
        if st.get("mocks"):
            cards["mocks"] = st["mocks"]
        if st.get("loader"):
            cards["loader"] = st["loader"]
        return cards

    def transfer(self, data: Any = None, source: Optional[str] = None, target: Optional[str] = None,
                 out: Optional[str] = None, device: Optional[str] = None, deep: Optional[bool] = None,
                 **kwargs) -> Dict[str, Any]:
        prev = self.apply_env()
        try:
            gb = self._gb()
            cfg = self.config
            src = source or cfg.default_source or "auto"
            tgt = target or cfg.default_target or "host"
            dev = device if device is not None else cfg.device
            d = cfg.deep if deep is None else bool(deep)
            td = dict(cfg.transfer_defaults or {})
            if "out" in td and out is None:
                out = td.get("out")
            res = gb.transfer_vram(data=data, source=src, target=tgt, out=out, device=dev, deep=d)
            res["runtime"] = {"name": cfg.name}
            return res
        finally:
            self.restore_env(prev)

    def save(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        return {"saved": True, "path": str(p), "config": self.config.to_dict()}

    def status(self) -> Dict[str, Any]:
        st = {"name": self.config.name, "created_at_ns": self.created_at_ns, "config": self.config.to_dict()}
        st["lumes"] = self.config.lumes
        if self.config.enable_mocks:
            st["mocks"] = {"enabled": True, "devices": list(self.config.mock_devices) if self.config.mock_devices else "default"}
            st["loader"] = self._mock_state.get("glow_stick_loader") or {"name": "glow_stick_loader", "attached": False, "glow_stick": True, "lumes": self.config.lumes}
        if self.config.force_load:
            st["force_load"] = {"enabled": True, "functions": list(self.config.force_load_functions) or "default", "loaded": self._force_loaded}
        if self.config.require_null_for_place_turn:
            st["place_turn"] = {"require_null": True, "config": self.config.place_turn_config}
        return st


def place_turn(turn: Any = None, slot: Any = None, runtime: Optional[CustomRuntime] = None, config: Any = None, **kwargs) -> Dict[str, Any]:
    """Module-level helper: place a turn; requires null slot when runtime/config enforces it."""
    rt = runtime or (CustomRuntime.from_config(config) if config is not None else CustomRuntime(RuntimeConfig(require_null_for_place_turn=True)))
    if not isinstance(rt, CustomRuntime):
        rt = CustomRuntime.from_config(config)
    return rt.place_turn(turn=turn, slot=slot, **kwargs)


def glow_stick_moc_to_loader(runtime: Optional[CustomRuntime] = None, config: Any = None, mock_devices: Optional[list] = None) -> Dict[str, Any]:
    """Attach ('glow stick') mock devices to the runtime loader."""
    rt = runtime or CustomRuntime.from_config(config)
    if mock_devices is not None:
        rt.config.mock_devices = list(mock_devices)
    rt.config.enable_mocks = True
    st: Dict[str, Any] = {"runtime": rt.config.name}
    st = rt._inject_mocks(st)
    st = rt._attach_glow_stick_loader(st, rt._mock_state.get("devices"))
    return {"ok": True, "loader": st.get("loader"), "mocks": st.get("mocks")}


def force_load(functions: Optional[list] = None, runtime: Optional[CustomRuntime] = None, config: Any = None) -> Dict[str, Any]:
    """Module-level helper: force-load runtime functions/mocks."""
    rt = runtime or CustomRuntime.from_config(config)
    if functions:
        rt.config.force_load_functions = list(functions)
    rt.config.force_load = True
    if rt.config.enable_mocks:
        rt._mock_state.setdefault("devices", list(rt.config.mock_devices) if rt.config.mock_devices else [])
    return rt.force_load(functions=functions)


def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mathlib_py runtime", description="Custom runtime for GPU probe/detect/host-staged transfer.")
    ap.add_argument("cmd", nargs="?", default="status", choices=["status", "probe", "detect", "transfer", "init", "save", "force-load", "place-turn"], help="runtime command")
    ap.add_argument("--config", help="Runtime config JSON string or @path/path")
    ap.add_argument("--save", help="Path to save runtime config JSON")
    ap.add_argument("--name", help="Runtime name")
    ap.add_argument("--prefer", choices=["vulkan", "cuda"], help="Preferred backend")
    ap.add_argument("--source", help="Default/transfer source")
    ap.add_argument("--target", help="Default/transfer target")
    ap.add_argument("--device", help="CUDA device")
    ap.add_argument("--deep", action="store_true", help="Deep probe")
    ap.add_argument("--leviathan-path")
    ap.add_argument("--gotlet-path")
    ap.add_argument("--gotlet-nvml-sim")
    ap.add_argument("--gotlet-card-specs")
    ap.add_argument("--nm-resistor-path")
    ap.add_argument("--nvidia-downloads-path")
    ap.add_argument("--optimas-path")
    ap.add_argument("--data", help="Transfer payload JSON/array or @file")
    ap.add_argument("--file", help="Transfer payload file")
    ap.add_argument("--out", help="Staged output path")
    ap.add_argument("--dtype")
    ap.add_argument("--shape")
    ap.add_argument("--enable-mocks", action="store_true", help="Enable mock devices in runtime")
    ap.add_argument("--mock-devices", help="JSON array or @path of mock device specs")
    ap.add_argument("--force-load", action="store_true", help="Force-load configured runtime functions")
    ap.add_argument("--force-load-functions", help="JSON array or comma-separated function names to force-load")
    ap.add_argument("--require-null-place-turn", action=argparse.BooleanOptionalAction, default=None, help="Require null slot to place turns")
    ap.add_argument("--place-turn-config", help="JSON object or @path for place_turn config")
    ap.add_argument("--turn", help="Turn payload (JSON/string) for place-turn command")
    ap.add_argument("--slot", help="Turn slot (must be null/None when require-null enabled; pass literal 'null' for valid placement)")
    ns = ap.parse_args(argv)

    cfg = RuntimeConfig.from_json(ns.config) if ns.config else RuntimeConfig()
    if ns.name: cfg.name = ns.name
    if ns.prefer: cfg.prefer = ns.prefer
    if ns.source: cfg.default_source = ns.source
    if ns.target: cfg.default_target = ns.target
    if ns.device: cfg.device = ns.device
    if ns.deep: cfg.deep = True
    for attr, val in [
        ("leviathan_path", ns.leviathan_path), ("gotlet_path", ns.gotlet_path),
        ("gotlet_nvml_sim", ns.gotlet_nvml_sim), ("gotlet_card_specs", ns.gotlet_card_specs),
        ("nm_resistor_path", ns.nm_resistor_path), ("nvidia_downloads_path", ns.nvidia_downloads_path),
        ("optimas_path", ns.optimas_path),
    ]:
        if val:
            setattr(cfg, attr, val)
    if ns.enable_mocks:
        cfg.enable_mocks = True
    if ns.mock_devices:
        try:
            cfg.mock_devices = json.loads(ns.mock_devices) if not str(ns.mock_devices).startswith("@") else json.loads(Path(str(ns.mock_devices)[1:]).read_text(encoding="utf-8"))
        except Exception:
            cfg.mock_devices = [{"name": ns.mock_devices}]
    if ns.force_load:
        cfg.force_load = True
    if ns.force_load_functions:
        try:
            if str(ns.force_load_functions).startswith("@"):
                cfg.force_load_functions = json.loads(Path(str(ns.force_load_functions)[1:]).read_text(encoding="utf-8"))
            else:
                cfg.force_load_functions = json.loads(ns.force_load_functions)
        except Exception:
            cfg.force_load_functions = [x.strip() for x in str(ns.force_load_functions).split(",") if x.strip()]
    if ns.require_null_place_turn is not None:
        cfg.require_null_for_place_turn = bool(ns.require_null_place_turn)
    if ns.place_turn_config:
        try:
            if str(ns.place_turn_config).startswith("@"):
                cfg.place_turn_config = json.loads(Path(str(ns.place_turn_config)[1:]).read_text(encoding="utf-8"))
            else:
                cfg.place_turn_config = json.loads(ns.place_turn_config)
        except Exception:
            cfg.place_turn_config = {"raw": ns.place_turn_config}

    rt = CustomRuntime(cfg)
    cmd = ns.cmd
    if cmd == "init":
        if ns.save:
            rt.save(ns.save)
        print(json.dumps({"ok": True, "runtime": rt.status(), "saved": ns.save}, indent=2))
        return 0
    if cmd == "save":
        if not ns.save:
            print(json.dumps({"ok": False, "error": "--save required"}, indent=2))
            return 1
        print(json.dumps(rt.save(ns.save), indent=2))
        return 0
    if cmd == "status":
        print(json.dumps(rt.status(), indent=2))
        return 0
    if cmd == "force-load":
        res = rt.force_load()
        if ns.save:
            rt.save(ns.save)
        print(json.dumps(res, indent=2, default=str))
        return 0
    if cmd == "place-turn":
        slot_val: Any = None
        if ns.slot is not None:
            s = str(ns.slot)
            if s.lower() in ("null", "none"):
                slot_val = None
            else:
                try:
                    slot_val = json.loads(s)
                except Exception:
                    slot_val = s
        turn_val: Any = ns.turn
        if ns.turn is not None:
            try:
                turn_val = json.loads(ns.turn)
            except Exception:
                turn_val = ns.turn
        res = rt.place_turn(turn=turn_val, slot=slot_val)
        if ns.save:
            rt.save(ns.save)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if cmd == "probe":
        res = rt.probe(deep=ns.deep or cfg.deep)
        if ns.save:
            rt.save(ns.save)
        print(json.dumps(res, indent=2, default=str))
        return 0
    if cmd == "detect":
        res = rt.detect_cards(deep=ns.deep or cfg.deep)
        if ns.save:
            rt.save(ns.save)
        print(json.dumps(res, indent=2, default=str))
        return 0
    if cmd == "transfer":
        payload = None
        if ns.file:
            payload = {"file": ns.file}
            if ns.dtype: payload["dtype"] = ns.dtype
            if ns.shape: payload["shape"] = [int(x) for x in ns.shape.split(",") if x.strip()]
        elif ns.data:
            from .gpu_backends import _load_json_arg
            payload = _load_json_arg(ns.data)
        res = rt.transfer(data=payload, source=ns.source, target=ns.target, out=ns.out, device=ns.device, deep=ns.deep)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    print(json.dumps({"ok": False, "error": f"unknown cmd {cmd}"}))
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli(os.sys.argv[1:]))
