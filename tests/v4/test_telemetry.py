"""GPU telemetry against a fake pynvml (the NVML path used on real NVIDIA machines)."""

import sys
import time
import types

import pytest

from suite_v4.telemetry import GPUTelemetry, describe_gpus, get_system_info


def _fake_pynvml(power_mw=(250_000, 260_000), reasons=0x4):
    nv = types.ModuleType("pynvml")
    nv.NVML_TEMPERATURE_GPU = 0
    nv.NVML_CLOCK_SM = 1
    nv.NVML_CLOCK_MEM = 2
    energy = {0: 1_000_000, 1: 2_000_000}  # mJ

    nv.nvmlInit = lambda: None
    nv.nvmlShutdown = lambda: None
    nv.nvmlDeviceGetCount = lambda: 2
    nv.nvmlDeviceGetHandleByIndex = lambda i: i
    nv.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=90 + h)
    nv.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=80 * 2**30, total=96 * 2**30)
    nv.nvmlDeviceGetTemperature = lambda h, s: 60 + h
    nv.nvmlDeviceGetPowerUsage = lambda h: power_mw[h]
    nv.nvmlDeviceGetClockInfo = lambda h, c: 1800 if c == 1 else 13800
    nv.nvmlDeviceGetMaxClockInfo = lambda h, c: 3090
    nv.nvmlDeviceGetCurrentClocksEventReasons = lambda h: reasons

    def get_energy(h):
        energy[h] += 30_000  # +30 J per read
        return energy[h]

    nv.nvmlDeviceGetTotalEnergyConsumption = get_energy
    nv.nvmlDeviceGetEnforcedPowerLimit = lambda h: 300_000
    nv.nvmlDeviceGetPowerManagementDefaultLimit = lambda h: 600_000
    nv.nvmlDeviceGetName = lambda h: "NVIDIA RTX PRO 6000 Blackwell"
    nv.nvmlSystemGetDriverVersion = lambda: "580.82.07"
    nv.nvmlSystemGetCudaDriverVersion_v2 = lambda: 13000
    return nv


def test_nvml_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    gpu = GPUTelemetry(poll_interval=0.01)
    assert gpu.backend_name == "nvml"
    info = get_system_info(gpu.static_info())
    assert info["gpu_name"] == "2x NVIDIA RTX PRO 6000 Blackwell"
    assert info["power_limits_w"] == [300.0, 300.0] and info["default_power_limits_w"] == [600.0, 600.0]
    assert info["cuda_version"] == "13.0" and info["total_vram_gb"] == pytest.approx(192.0)

    gpu.start()
    t0 = time.perf_counter()
    e0 = gpu.energy_joules()
    time.sleep(0.15)
    e1 = gpu.energy_joules()
    t1 = time.perf_counter()
    gpu.stop()
    assert e1 - e0 == pytest.approx(60.0)  # two GPUs, +30 J each
    stats = gpu.window_stats(t0, t1)
    assert stats["avg_power"] == pytest.approx(510.0)  # summed over GPUs, in watts
    assert stats["avg_gpu_util"] == pytest.approx(90.5)  # averaged over GPUs
    assert stats["max_temperature"] == 61 and stats["avg_gpu_clock"] == 1800
    assert stats["power_cap_pct"] == pytest.approx(100.0) and stats["thermal_throttle_pct"] == 0.0
    assert stats["power_time_avg"] == pytest.approx(510.0)
    tl = gpu.timeline(t0 - 1, bin_seconds=0.05)
    assert tl["t"] and tl["power_draw"][0] == pytest.approx(510.0)


def test_gpu_subset_and_disabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(reasons=0))
    gpu = GPUTelemetry(gpu_indices=[1])
    assert gpu.static_info()["gpu_indices"] == [1]
    assert gpu.backend.sample()["power_draw"] == pytest.approx(260.0)
    assert gpu.backend.sample()["power_capped"] is False
    assert not GPUTelemetry(enabled=False).available


def test_describe_gpus():
    assert describe_gpus(["A", "A", "B"]) == "2x A + B"
    assert describe_gpus([]) is None
