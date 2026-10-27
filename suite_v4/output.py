"""Result files: JSON package (checkpointed after every test), CSVs and the LLM-Inference-Bench CSV."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Column order of LLM-Inference-Bench's results CSVs (Plots/All_results.csv + power columns),
# followed by the TTFT columns of its TTFT/ITL figure data.
ARGONNE_COLUMNS = [
    "Hardware",
    "Num of Hardware",
    "Framework",
    "Model",
    "Input Output Length",
    "Batch Size",
    "Latency",
    "Throughput",
    "avg_power",
    "Throughput_per_watt_avg",
    "Input Length",
    "Output Length",
    "TTFT Latency",
    "end_latency",
]


def output_paths(output_dir: str, stem: str) -> Dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "json": out / f"{stem}.json",
        "csv": out / f"{stem}.csv",
        "argonne_csv": out / f"{stem}_llm-inference-bench.csv",
        "requests_csv": out / f"{stem}_requests.csv",
        "png": out / f"{stem}.png",
    }


def write_json(path: Path, package: Dict[str, Any]) -> None:
    """Atomic write so an interrupted run never leaves a truncated file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(package, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_results_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results])
    df.to_csv(path, index=False)


def write_argonne_csv(path: Path, results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    system = metadata.get("system_info", {})
    server = metadata.get("server_info", {})
    cfg = metadata.get("configuration", {})
    names = system.get("gpu_names") or []
    hardware = cfg.get("hardware") or (names[0] if names else system.get("gpu_name")) or "Unknown"
    tp = server.get("tensor_parallel") or 1
    pp = server.get("pipeline_parallel") or 1
    num_hw = cfg.get("num_gpus") or (tp * pp if server.get("tensor_parallel") else system.get("gpu_count")) or 1
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ARGONNE_COLUMNS)
        for r in results:
            if r.get("status") not in ("ok", "partial"):
                continue
            ttft = r.get("ttft_mean_ms")
            writer.writerow(
                [
                    hardware,
                    num_hw,
                    server.get("framework") or "vLLM",
                    server.get("model_name") or metadata.get("model"),
                    r["input_len"],
                    r["batch_size"],
                    r["latency_avg"],
                    r["throughput"],
                    r.get("avg_power", ""),  # watts (the paper's power CSVs store raw NVML milliwatts)
                    r.get("throughput_per_watt", ""),
                    r["input_len"],
                    r["output_len"],
                    ttft / 1000.0 if ttft is not None else "",
                    r["latency_avg"],
                ]
            )


def write_requests_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)
