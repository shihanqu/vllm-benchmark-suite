#!/usr/bin/env python3
"""
Stand-in for `nvidia-smi` (install as an executable named nvidia-smi on PATH) that reports two
power-limited GPUs whose power, clocks and temperature follow the mock server's load
(read from MOCK_GPU_STATE). Supports the queries the v4 telemetry module issues.
"""

import json
import os
import sys
import time

NUM_GPUS = 2
LIMIT_W = 300.0
IDLE_W = 38.0
MAX_CLOCK = 3090.0


def load_state():
    path = os.environ.get("MOCK_GPU_STATE")
    try:
        with open(path) as f:
            state = json.load(f)
        if time.time() - state.get("t", 0) > 2.0 and state.get("running", 0) == 0:
            return 0.0, 0.0
        return float(state.get("load", 0.0)), float(state.get("kv", 0.0))
    except Exception:
        return 0.0, 0.0


def gpu_values(index: int):
    load, kv = load_state()
    demand = IDLE_W + (560.0 - IDLE_W) * load  # unconstrained draw would exceed the limit
    capped = demand > LIMIT_W
    power = min(demand, LIMIT_W) - index * 3.0 + (2.0 if load else 0.0)
    clock = MAX_CLOCK * (0.62 if capped else (0.25 + 0.7 * load if load else 0.06))
    temp = 33 + power * 0.11 + index
    mem_total = 97887
    mem_used = 1200 + 84000 * 0.9 + 400 * kv  # vLLM preallocates the KV cache
    return {
        "index": str(index),
        "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "utilization.gpu": f"{min(100, 100 * load * 1.1 if load else 0):.0f}",
        "memory.used": f"{mem_used:.0f}",
        "memory.total": str(mem_total),
        "temperature.gpu": f"{temp:.0f}",
        "power.draw": f"{power:.2f}",
        "clocks.gr": f"{clock:.0f}",
        "clocks.sm": f"{clock:.0f}",
        "clocks.mem": "13801",
        "power.limit": f"{LIMIT_W:.2f}",
        "power.default_limit": "600.00",
        "clocks.max.sm": f"{MAX_CLOCK:.0f}",
        "driver_version": "580.82.07",
        "clocks_event_reasons.sw_power_cap": "Active" if capped else "Not Active",
        "clocks_event_reasons.sw_thermal_slowdown": "Not Active",
        "clocks_event_reasons.hw_slowdown": "Not Active",
    }


def main(argv):
    query = next((a.split("=", 1)[1] for a in argv if a.startswith("--query-gpu=")), None)
    if query is None:
        print("+-----------------------------------------------------------------------------+")
        print("| NVIDIA-SMI 580.82.07    Driver Version: 580.82.07    CUDA Version: 13.0     |")
        print("+-----------------------------------------------------------------------------+")
        return 0
    indices = list(range(NUM_GPUS))
    if "-i" in argv:
        indices = [int(x) for x in argv[argv.index("-i") + 1].split(",")]
    fields = query.split(",")
    for i in indices:
        values = gpu_values(i)
        if any(f not in values for f in fields):
            print(f"Field '{next(f for f in fields if f not in values)}' is not a valid field to query.", file=sys.stderr)
            return 2
        print(", ".join(values[f] for f in fields))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
