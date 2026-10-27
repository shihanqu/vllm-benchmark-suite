"""
GPU and host telemetry.

NVML (nvidia-ml-py) is preferred, as in LLM-Inference-Bench's power_utils.py: it is cheap to
poll and exposes the hardware energy counter, the enforced power limit and clock throttle
reasons. When NVML is unavailable the monitor falls back to polling nvidia-smi like v3.
GPU indices are NVML / nvidia-smi indices (PCI order), not CUDA_VISIBLE_DEVICES order.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

# NVML clock event / throttle reason bits (stable across driver versions)
_REASON_SW_POWER_CAP = 0x4
_REASON_HW_SLOWDOWN = 0x8
_REASON_SW_THERMAL = 0x20
_REASON_HW_THERMAL = 0x40
_REASON_HW_POWER_BRAKE = 0x80
_POWER_REASONS = _REASON_SW_POWER_CAP | _REASON_HW_POWER_BRAKE
_THERMAL_REASONS = _REASON_SW_THERMAL | _REASON_HW_THERMAL | _REASON_HW_SLOWDOWN

_SMI_BASE_FIELDS = [
    "index",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.gr",
    "clocks.mem",
]
# Newer drivers call these clocks_event_reasons; older ones clocks_throttle_reasons.
_SMI_REASON_FIELD_SETS = [
    ["clocks_event_reasons.sw_power_cap", "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_slowdown"],
    ["clocks_throttle_reasons.sw_power_cap", "clocks_throttle_reasons.sw_thermal_slowdown", "clocks_throttle_reasons.hw_slowdown"],
    [],
]


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def describe_gpus(names: Sequence[str]) -> Optional[str]:
    """'2x NVIDIA RTX PRO 6000' style description, grouping identical GPUs like v3."""
    names = [n for n in names if n]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return " + ".join(f"{c}x {n}" if c > 1 else n for n, c in Counter(names).items())


class _NvmlBackend:
    name = "nvml"

    def __init__(self, gpu_indices: Optional[Sequence[int]]):
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        self.indices = list(gpu_indices) if gpu_indices else list(range(count))
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.indices]
        if not self.handles:
            raise RuntimeError("NVML found no GPUs")
        self._reasons_fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None
        )

    def _safe(self, fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def sample(self) -> Optional[Dict[str, Any]]:
        nv = self.nv
        per_gpu = []
        for h in self.handles:
            util = self._safe(nv.nvmlDeviceGetUtilizationRates, h)
            mem = self._safe(nv.nvmlDeviceGetMemoryInfo, h)
            power = self._safe(nv.nvmlDeviceGetPowerUsage, h)
            reasons = self._safe(self._reasons_fn, h) if self._reasons_fn else None
            per_gpu.append(
                {
                    "gpu_util": float(util.gpu) if util is not None else None,
                    "mem_used": mem.used / 2**20 if mem is not None else None,
                    "mem_total": mem.total / 2**20 if mem is not None else None,
                    "temperature": _to_float(self._safe(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)),
                    "power_draw": power / 1000.0 if power is not None else None,
                    "gpu_clock": _to_float(self._safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)),
                    "mem_clock": _to_float(self._safe(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM)),
                    "reasons": reasons,
                }
            )
        return _aggregate(per_gpu)

    def energy_joules(self) -> Optional[float]:
        total = 0.0
        for h in self.handles:
            mj = self._safe(self.nv.nvmlDeviceGetTotalEnergyConsumption, h)
            if mj is None:
                return None
            total += mj / 1000.0
        return total

    def static_info(self) -> Dict[str, Any]:
        nv = self.nv
        names, vram, limits, default_limits, max_sm = [], [], [], [], []
        for h in self.handles:
            name = self._safe(nv.nvmlDeviceGetName, h)
            names.append(name.decode() if isinstance(name, bytes) else name)
            mem = self._safe(nv.nvmlDeviceGetMemoryInfo, h)
            vram.append(mem.total / 2**30 if mem is not None else None)
            lim = self._safe(nv.nvmlDeviceGetEnforcedPowerLimit, h)
            limits.append(lim / 1000.0 if lim is not None else None)
            dlim = self._safe(nv.nvmlDeviceGetPowerManagementDefaultLimit, h)
            default_limits.append(dlim / 1000.0 if dlim is not None else None)
            max_sm.append(_to_float(self._safe(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_SM)))
        driver = self._safe(nv.nvmlSystemGetDriverVersion)
        cuda_fn = getattr(nv, "nvmlSystemGetCudaDriverVersion_v2", None) or getattr(nv, "nvmlSystemGetCudaDriverVersion", None)
        cuda = self._safe(cuda_fn) if cuda_fn else None
        return {
            "gpu_indices": self.indices,
            "gpu_names": names,
            "gpu_count": len(self.handles),
            "total_vram_gb": sum(v for v in vram if v) or None,
            "power_limits_w": limits,
            "default_power_limits_w": default_limits,
            "max_sm_clock_mhz": max_sm,
            "driver_version": driver.decode() if isinstance(driver, bytes) else driver,
            "cuda_version": f"{cuda // 1000}.{(cuda % 1000) // 10}" if isinstance(cuda, int) else None,
        }

    def close(self) -> None:
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass


class _SmiBackend:
    name = "nvidia-smi"

    def __init__(self, gpu_indices: Optional[Sequence[int]]):
        self.indices = list(gpu_indices) if gpu_indices else None
        self.fields = None
        for reason_fields in _SMI_REASON_FIELD_SETS:
            fields = _SMI_BASE_FIELDS + reason_fields
            if self._query(fields) is not None:
                self.fields = fields
                break
        if self.fields is None:
            raise RuntimeError("nvidia-smi query failed")

    def _query(self, fields: List[str]) -> Optional[List[Dict[str, str]]]:
        cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
        if self.indices:
            cmd += ["-i", ",".join(str(i) for i in self.indices)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        except Exception:
            return None
        if result.returncode != 0:
            return None
        rows = []
        for line in result.stdout.strip().splitlines():
            values = [v.strip() for v in line.split(",")]
            if len(values) == len(fields):
                rows.append(dict(zip(fields, values)))
        return rows or None

    def sample(self) -> Optional[Dict[str, Any]]:
        rows = self._query(self.fields)
        if not rows:
            return None
        per_gpu = []
        for r in rows:
            reasons = 0
            if any(r.get(f, "").lower() == "active" for f in self.fields if f.endswith("sw_power_cap")):
                reasons |= _REASON_SW_POWER_CAP
            if any(r.get(f, "").lower() == "active" for f in self.fields if "thermal" in f or f.endswith("hw_slowdown")):
                reasons |= _REASON_SW_THERMAL
            per_gpu.append(
                {
                    "gpu_util": _to_float(r.get("utilization.gpu")),
                    "mem_used": _to_float(r.get("memory.used")),
                    "mem_total": _to_float(r.get("memory.total")),
                    "temperature": _to_float(r.get("temperature.gpu")),
                    "power_draw": _to_float(r.get("power.draw")),
                    "gpu_clock": _to_float(r.get("clocks.gr")),
                    "mem_clock": _to_float(r.get("clocks.mem")),
                    "reasons": reasons if len(self.fields) > len(_SMI_BASE_FIELDS) else None,
                }
            )
        return _aggregate(per_gpu)

    def energy_joules(self) -> Optional[float]:
        return None

    def static_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        rows = self._query(["index", "name", "memory.total", "power.limit", "power.default_limit", "clocks.max.sm", "driver_version"])
        if rows:
            info["gpu_indices"] = [int(r["index"]) for r in rows]
            info["gpu_names"] = [r["name"] for r in rows]
            info["gpu_count"] = len(rows)
            info["total_vram_gb"] = sum(_to_float(r["memory.total"]) or 0 for r in rows) / 1024 or None
            info["power_limits_w"] = [_to_float(r["power.limit"]) for r in rows]
            info["default_power_limits_w"] = [_to_float(r["power.default_limit"]) for r in rows]
            info["max_sm_clock_mhz"] = [_to_float(r["clocks.max.sm"]) for r in rows]
            info["driver_version"] = rows[0]["driver_version"]
        try:
            out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3).stdout
            match = re.search(r"CUDA Version: ([\d.]+)", out)
            info["cuda_version"] = match.group(1) if match else None
        except Exception:
            info["cuda_version"] = None
        return info

    def close(self) -> None:
        pass


def _aggregate(per_gpu: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Combine GPUs like v3: mean utilization/temperature/clocks, summed memory and power."""
    if not per_gpu:
        return None

    def vals(key):
        return [g[key] for g in per_gpu if g.get(key) is not None]

    def avg(key):
        v = vals(key)
        return mean(v) if v else None

    def total(key):
        v = vals(key)
        return sum(v) if v else None

    reasons = [g["reasons"] for g in per_gpu if g.get("reasons") is not None]
    temps = vals("temperature")
    return {
        "gpu_util": avg("gpu_util"),
        "mem_used": total("mem_used"),
        "mem_total": total("mem_total"),
        "temperature": avg("temperature"),
        "temperature_max": max(temps) if temps else None,
        "power_draw": total("power_draw"),
        "gpu_clock": avg("gpu_clock"),
        "mem_clock": avg("mem_clock"),
        "power_capped": any(r & _POWER_REASONS for r in reasons) if reasons else None,
        "thermal_throttled": any(r & _THERMAL_REASONS for r in reasons) if reasons else None,
        "num_gpus": len(per_gpu),
    }


class GPUTelemetry:
    """Background GPU sampler with per-window statistics and energy accounting."""

    def __init__(self, poll_interval: float = 0.1, gpu_indices: Optional[Sequence[int]] = None, enabled: bool = True):
        self.poll_interval = poll_interval
        self.samples: List[Dict[str, Any]] = []
        self.backend = None
        self.error: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        if not enabled:
            self.error = "disabled"
            return
        errors = []
        for backend_cls in (_NvmlBackend, _SmiBackend):
            try:
                self.backend = backend_cls(gpu_indices)
                break
            except Exception as e:
                errors.append(f"{backend_cls.name}: {e}")
        if self.backend is None:
            self.error = "; ".join(errors)

    @property
    def available(self) -> bool:
        return self.backend is not None

    @property
    def backend_name(self) -> Optional[str]:
        return self.backend.name if self.backend else None

    def start(self) -> None:
        if not self.backend or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.backend:
            self.backend.close()

    def _loop(self) -> None:
        last_error = 0.0
        while self._running:
            t0 = time.perf_counter()
            try:
                s = self.backend.sample()
                if s:
                    s["t"] = time.perf_counter()
                    self.samples.append(s)
            except Exception as e:
                if time.time() - last_error > 10:
                    print(f"[WARNING] GPU monitoring error: {e}", file=sys.stderr)
                    last_error = time.time()
            time.sleep(max(0.0, self.poll_interval - (time.perf_counter() - t0)))

    def latest(self) -> Optional[Dict[str, Any]]:
        return self.samples[-1] if self.samples else None

    def energy_joules(self) -> Optional[float]:
        """Cumulative hardware energy counter summed over monitored GPUs (NVML only)."""
        if not self.backend:
            return None
        try:
            return self.backend.energy_joules()
        except Exception:
            return None

    def static_info(self) -> Dict[str, Any]:
        if not self.backend:
            return {}
        try:
            return self.backend.static_info()
        except Exception:
            return {}

    def samples_between(self, t0: float, t1: float) -> List[Dict[str, Any]]:
        return [s for s in list(self.samples) if t0 <= s["t"] <= t1]

    def window_stats(self, t0: float, t1: float) -> Optional[Dict[str, Any]]:
        """v3-compatible averages/peaks for [t0, t1] plus throttle residency and integrated energy."""
        window = self.samples_between(t0, t1)
        if not window:
            # Very short windows can fall between two polls; use the nearest sample.
            prior = [s for s in list(self.samples) if s["t"] <= t1]
            window = prior[-1:] if prior else []
        if not window:
            return None

        def series(key):
            return [s[key] for s in window if s.get(key) is not None]

        stats: Dict[str, Any] = {"samples": len(window)}
        for key, name in (
            ("gpu_util", "gpu_util"),
            ("mem_used", "mem_used"),
            ("temperature", "temperature"),
            ("power_draw", "power"),
            ("gpu_clock", "gpu_clock"),
            ("mem_clock", "mem_clock"),
        ):
            v = series(key)
            if v:
                stats[f"avg_{name}"] = mean(v)
                stats[f"max_{name}"] = max(v)
                if key == "gpu_clock":
                    stats["min_gpu_clock"] = min(v)
        temps_max = series("temperature_max")
        if temps_max:
            stats["max_temperature"] = max(temps_max)
        for flag, name in (("power_capped", "power_cap_pct"), ("thermal_throttled", "thermal_throttle_pct")):
            v = series(flag)
            if v:
                stats[name] = 100.0 * sum(1 for x in v if x) / len(v)
        # Time-weighted mean power (trapezoidal), used for energy when there is no energy counter
        pts = [(s["t"], s["power_draw"]) for s in window if s.get("power_draw") is not None]
        if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
            integral = sum((t2 - t1_) * (p1 + p2) / 2 for (t1_, p1), (t2, p2) in zip(pts, pts[1:]))
            stats["power_time_avg"] = integral / (pts[-1][0] - pts[0][0])
        elif pts:
            stats["power_time_avg"] = pts[0][1]
        return stats

    def measure_idle_power(self, seconds: float) -> Optional[float]:
        """Mean power over the next `seconds` while nothing is running."""
        if not self.backend or seconds <= 0:
            return None
        t0 = time.perf_counter()
        time.sleep(seconds)
        stats = self.window_stats(t0, time.perf_counter())
        return stats.get("avg_power") if stats else None

    def timeline(self, t_origin: float, bin_seconds: float = 1.0) -> Dict[str, List[Optional[float]]]:
        """Samples averaged into fixed bins, relative to t_origin, for the run timeline chart."""
        keys = ("power_draw", "gpu_clock", "temperature", "gpu_util", "mem_used")
        bins: Dict[int, List[Dict[str, Any]]] = {}
        for s in list(self.samples):
            if s["t"] < t_origin:
                continue
            bins.setdefault(int((s["t"] - t_origin) / bin_seconds), []).append(s)
        out: Dict[str, List[Optional[float]]] = {"t": []}
        for k in keys:
            out[k] = []
        for b in sorted(bins):
            group = bins[b]
            out["t"].append(round((b + 0.5) * bin_seconds, 3))
            for k in keys:
                v = [s[k] for s in group if s.get(k) is not None]
                out[k].append(round(mean(v), 3) if v else None)
        return out


def get_system_info(gpu_static: Dict[str, Any]) -> Dict[str, Any]:
    """Host + GPU description (v3 SystemInfo fields plus power limits)."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_version": gpu_static.get("cuda_version"),
        "driver_version": gpu_static.get("driver_version"),
        "gpu_name": describe_gpus(gpu_static.get("gpu_names") or []),
        "gpu_names": gpu_static.get("gpu_names"),
        "gpu_count": gpu_static.get("gpu_count"),
        "gpu_indices": gpu_static.get("gpu_indices"),
        "total_vram_gb": gpu_static.get("total_vram_gb"),
        "power_limits_w": gpu_static.get("power_limits_w"),
        "default_power_limits_w": gpu_static.get("default_power_limits_w"),
        "max_sm_clock_mhz": gpu_static.get("max_sm_clock_mhz"),
        "timestamp": datetime.now().isoformat(),
    }
    return info
