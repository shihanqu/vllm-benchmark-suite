"""
Per-test aggregation.

LLM-Inference-Bench definitions (kept verbatim so numbers are comparable to the paper):
  Latency    = mean wall time of one batch (all B requests complete), over measured iterations
  Throughput = B * (input_len + output_len) / Latency      [total tokens per second]
v4 adds measured streaming metrics (TTFT, TPOT, ITL), generation throughput (v3's tokens/sec),
energy from the GPU energy counter, and server-side internals.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import numpy as np

from .engines import BatchResult

ARGONNE_PERCENTILES = (10, 25, 50, 75, 90, 99)


def _pct(values: List[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def _ms(value: Optional[float]) -> Optional[float]:
    return value * 1000.0 if value is not None else None


def summarize_requests(iterations: List[BatchResult]) -> Dict[str, Any]:
    """Streaming metrics over every successful request in the measured iterations."""
    ok = [r for it in iterations for r in it.ok_records]
    out: Dict[str, Any] = {}
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok if r.e2e is not None]
    tpots = [
        (r.e2e - r.ttft) / (r.completion_tokens - 1)
        for r in ok
        if r.ttft is not None and r.e2e is not None and (r.completion_tokens or 0) > 1
    ]
    itls = [x for r in ok for x in r.itls]

    if ttfts:
        out["ttft_mean_ms"] = _ms(mean(ttfts))
        for q in (50, 90, 99):
            out[f"ttft_p{q}_ms"] = _ms(_pct(ttfts, q))
        out["ttft_source"] = "stream"
    if tpots:
        out["tpot_mean_ms"] = _ms(mean(tpots))
        for q in (50, 90, 99):
            out[f"tpot_p{q}_ms"] = _ms(_pct(tpots, q))
        out["decode_speed_per_user"] = 1.0 / mean(tpots)
    if itls:
        out["itl_mean_ms"] = _ms(mean(itls))
        for q in (50, 90, 99):
            out[f"itl_p{q}_ms"] = _ms(_pct(itls, q))
    if e2es:
        out["e2e_mean"] = mean(e2es)
        out["e2e_std"] = pstdev(e2es) if len(e2es) > 1 else 0.0
        for q in (50, 90, 99):
            out[f"e2e_p{q}"] = _pct(e2es, q)
    chunks = sum(r.chunks for r in ok)
    if chunks:
        out["tokens_per_chunk"] = sum(r.completion_tokens or 0 for r in ok) / chunks

    # Effective prefill throughput: prompt tokens of the batch / time until the last request's
    # first token. For B=1 this is the plain single-request prefill speed.
    prefill = []
    for it in iterations:
        recs = [r for r in it.ok_records if r.ttft is not None]
        if recs:
            last_first_token = max(r.start + r.ttft for r in recs) - it.start
            tokens = sum(r.prompt_tokens or 0 for r in recs)
            if last_first_token > 0 and tokens:
                prefill.append(tokens / last_first_token)
    if prefill:
        out["prefill_throughput"] = mean(prefill)
    return out


def aggregate_test(
    input_len: int,
    output_len: int,
    batch_size: int,
    iterations: List[BatchResult],
    gpu_stats: Optional[Dict[str, Any]] = None,
    energy_counter_j: Optional[float] = None,
    window_seconds: Optional[float] = None,
    server_stats: Optional[Dict[str, Any]] = None,
    probe_latency: Optional[float] = None,
    idle_power: Optional[float] = None,
) -> Dict[str, Any]:
    """One results row for a (input_len, output_len, batch_size) test."""
    row: Dict[str, Any] = {
        "input_len": input_len,
        "output_len": output_len,
        "batch_size": batch_size,
        "iterations": len(iterations),
    }
    records = [r for it in iterations for r in it.records]
    ok = [r for r in records if r.ok]
    row["requests"] = len(records)
    row["successful"] = len(ok)
    row["failed"] = len(records) - len(ok)
    row["success_rate"] = 100.0 * len(ok) / len(records) if records else 0.0
    errors = [r.error for r in records if r.error]
    if errors:
        row["error_sample"] = errors[0]
    if not ok:
        row["status"] = "failed"
        return row
    row["status"] = "ok" if not errors else "partial"

    # LLM-Inference-Bench batch latency and throughput
    latencies = [it.latency for it in iterations]
    row["batch_latencies"] = [round(x, 6) for x in latencies]
    row["latency_avg"] = mean(latencies)
    row["latency_std"] = pstdev(latencies) if len(latencies) > 1 else 0.0
    row["latency_min"] = min(latencies)
    row["latency_max"] = max(latencies)
    for q in ARGONNE_PERCENTILES:
        row[f"latency_p{q}"] = _pct(latencies, q)
    row["latency_cv"] = 100.0 * row["latency_std"] / row["latency_avg"] if row["latency_avg"] > 0 else 0.0

    in_tokens = [sum((r.prompt_tokens if r.prompt_tokens is not None else input_len) for r in it.ok_records) for it in iterations]
    out_tokens = [sum(r.completion_tokens or 0 for r in it.ok_records) for it in iterations]
    ok_per_it = [len(it.ok_records) for it in iterations]
    lat = row["latency_avg"]
    row["throughput"] = (mean(in_tokens) + mean(out_tokens)) / lat
    row["output_throughput"] = mean(out_tokens) / lat
    row["input_throughput"] = mean(in_tokens) / lat
    row["requests_per_second"] = mean(ok_per_it) / lat
    row["throughput_per_user"] = row["output_throughput"] / batch_size
    # Spread of generation throughput across iterations (shaded band on the main chart)
    per_it = [o / l for o, l in zip(out_tokens, latencies) if l > 0]
    row["output_throughput_p10"] = _pct(per_it, 10)
    row["output_throughput_p90"] = _pct(per_it, 90)

    row["prompt_tokens_mean"] = mean(r.prompt_tokens for r in ok if r.prompt_tokens is not None) if any(
        r.prompt_tokens is not None for r in ok
    ) else None
    row["completion_tokens_mean"] = mean(r.completion_tokens or 0 for r in ok)
    row["output_token_fidelity"] = 100.0 * row["completion_tokens_mean"] / output_len if output_len else None

    row.update(summarize_requests(iterations))
    if probe_latency is not None and "ttft_mean_ms" not in row:
        # Offline engine: LLM-Inference-Bench measures TTFT as the latency of a max_tokens=1 batch
        # and derives the per-token decode time from the full-batch latency.
        row["ttft_mean_ms"] = row["ttft_p50_ms"] = probe_latency * 1000.0
        row["ttft_source"] = "probe"
        if output_len > 1:
            tpot = max(0.0, lat - probe_latency) / (output_len - 1)
            row["tpot_mean_ms"] = row["tpot_p50_ms"] = tpot * 1000.0
            row["decode_speed_per_user"] = 1.0 / tpot if tpot > 0 else None
        if lat > 0:
            row["prefill_throughput"] = mean(in_tokens) / probe_latency if probe_latency > 0 else None
        row["e2e_mean"] = lat

    # GPU and energy
    if gpu_stats:
        row.update({k: v for k, v in gpu_stats.items() if k != "power_time_avg"})
        energy, source = None, None
        if energy_counter_j is not None and energy_counter_j > 0:
            energy, source = energy_counter_j, "nvml_energy_counter"
        elif gpu_stats.get("power_time_avg") and window_seconds:
            energy, source = gpu_stats["power_time_avg"] * window_seconds, "power_samples"
        total_out = sum(out_tokens)
        total_all = sum(in_tokens) + total_out
        avg_power = gpu_stats.get("avg_power")
        if energy:
            row["energy_joules"] = energy
            row["energy_source"] = source
            row["energy_watt_hours"] = energy / 3600.0
            row["window_seconds"] = window_seconds
            if total_out:
                row["energy_per_output_token"] = energy / total_out
                row["tokens_per_joule"] = total_out / energy
            if total_all:
                row["energy_per_total_token"] = energy / total_all
            if idle_power is not None and window_seconds:
                row["energy_above_idle_joules"] = max(0.0, energy - idle_power * window_seconds)
        if avg_power:
            # LLM-Inference-Bench "Throughput_per_watt_avg" (total tokens/s per watt)
            row["throughput_per_watt"] = row["throughput"] / avg_power
            row["tokens_per_watt"] = row["output_throughput"] / avg_power

    if server_stats:
        row.update(server_stats)

    # v3 column aliases so existing consolidation scripts can read v4 results
    row["context_length"] = input_len
    row["concurrent_users"] = batch_size
    row["tokens_per_second"] = row["output_throughput"]
    row["avg_latency"] = row.get("e2e_mean", lat)
    row["total_time"] = sum(latencies)
    if "energy_per_output_token" in row:
        row["energy_per_token"] = row["energy_per_output_token"]
    return row


def add_scaling_efficiency(rows: List[Dict[str, Any]]) -> None:
    """v3 batch scaling efficiency: throughput(B) / (throughput(B_min) * B / B_min) per shape."""
    by_shape: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("output_throughput"):
            by_shape.setdefault((r["input_len"], r["output_len"]), []).append(r)
    for group in by_shape.values():
        base = min(group, key=lambda r: r["batch_size"])
        per_seq = base["output_throughput"] / base["batch_size"]
        for r in group:
            r["batch_efficiency"] = 100.0 * r["output_throughput"] / (per_seq * r["batch_size"]) if per_seq else None
