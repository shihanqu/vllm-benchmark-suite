"""
Server-side metrics from the Prometheus /metrics endpoint (vLLM, with SGLang fallbacks).

Counters and histograms are snapshotted at the start and end of each measured window and
differenced; gauges (running/waiting requests, KV cache usage) are polled during the window.
Metric names changed across vLLM releases, so every quantity lists its known aliases.
"""

from __future__ import annotations

import re
import threading
import time
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

import requests

_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)")
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

COUNTERS = {
    "prefix_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
    "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "spec_accepted": ["vllm:spec_decode_num_accepted_tokens_total"],
    "spec_draft_tokens": ["vllm:spec_decode_num_draft_tokens_total"],
    "spec_drafts": ["vllm:spec_decode_num_drafts_total"],
    "prompt_tokens": ["vllm:prompt_tokens_total", "sglang:prompt_tokens_total"],
    "generation_tokens": ["vllm:generation_tokens_total", "sglang:generation_tokens_total"],
}

# Histograms: the window mean is delta(_sum) / delta(_count)
HISTOGRAMS = {
    "queue_time": ["vllm:request_queue_time_seconds"],
    "prefill_time": ["vllm:request_prefill_time_seconds"],
    "decode_time": ["vllm:request_decode_time_seconds"],
    "ttft": ["vllm:time_to_first_token_seconds", "sglang:time_to_first_token_seconds"],
    "e2e": ["vllm:e2e_request_latency_seconds", "sglang:e2e_request_latency_seconds"],
}

# Gauges: (aliases, how to combine label sets, scale to percent?)
GAUGES = {
    "running": (["vllm:num_requests_running", "sglang:num_running_reqs"], "sum", 1.0),
    "waiting": (["vllm:num_requests_waiting", "sglang:num_queue_reqs"], "sum", 1.0),
    "kv_usage": (["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"], "max", 100.0),
    "prefix_hit_rate": (["vllm:gpu_prefix_cache_hit_rate", "sglang:cache_hit_rate"], "max", 100.0),
    "spec_accept_length": (["sglang:spec_accept_length"], "max", 1.0),
}


class Snapshot:
    """One parsed /metrics scrape: values per metric name (all label sets) plus *_info labels."""

    def __init__(self, values: Dict[str, List[float]], info: Dict[str, Dict[str, str]], t: float):
        self.values = values
        self.info = info
        self.t = t

    def total(self, aliases: List[str], suffix: str = "") -> Optional[float]:
        for name in aliases:
            v = self.values.get(name + suffix)
            if v:
                return sum(v)
        return None

    def gauge(self, key: str) -> Optional[float]:
        aliases, how, scale = GAUGES[key]
        for name in aliases:
            v = self.values.get(name)
            if v:
                return (sum(v) if how == "sum" else max(v)) * scale
        return None

    def has_prefix(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.values)


def parse_prometheus(text: str) -> Snapshot:
    values: Dict[str, List[float]] = {}
    info: Dict[str, Dict[str, str]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, labels, raw = m.groups()
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        values.setdefault(name, []).append(value)
        if name.endswith("_info") and labels and name not in info:
            info[name] = dict(_LABEL_RE.findall(labels))
    return Snapshot(values, info, time.perf_counter())


def _delta(before: Snapshot, after: Snapshot, aliases: List[str], suffix: str = "") -> Optional[float]:
    a, b = after.total(aliases, suffix), before.total(aliases, suffix)
    if a is None or b is None:
        return None
    return max(0.0, a - b)


def derive_window(
    before: Optional[Snapshot], after: Optional[Snapshot], gauge_samples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Server-side statistics for one measured window."""
    out: Dict[str, Any] = {}
    if before is not None and after is not None:
        d = {k: _delta(before, after, aliases) for k, aliases in COUNTERS.items()}
        if d["prefix_queries"]:
            out["prefix_cache_hit_rate"] = 100.0 * (d["prefix_hits"] or 0.0) / d["prefix_queries"]
        if d["preemptions"] is not None:
            out["preemptions"] = d["preemptions"]
        if d["spec_draft_tokens"]:
            out["spec_acceptance_rate"] = 100.0 * (d["spec_accepted"] or 0.0) / d["spec_draft_tokens"]
        if d["spec_drafts"]:
            out["spec_mean_acceptance_length"] = 1.0 + (d["spec_accepted"] or 0.0) / d["spec_drafts"]
        if d["generation_tokens"] is not None:
            out["server_generation_tokens"] = d["generation_tokens"]
        if d["prompt_tokens"] is not None:
            out["server_prompt_tokens"] = d["prompt_tokens"]
        for key, aliases in HISTOGRAMS.items():
            s = _delta(before, after, aliases, "_sum")
            c = _delta(before, after, aliases, "_count")
            if s is not None and c:
                out[f"server_{key}_mean"] = s / c

    for key in ("running", "waiting", "kv_usage", "prefix_hit_rate", "spec_accept_length"):
        series = [g[key] for g in gauge_samples if g.get(key) is not None]
        if not series:
            continue
        if key == "kv_usage":
            out["kv_cache_usage_peak"] = max(series)
            out["kv_cache_usage_mean"] = mean(series)
        elif key in ("running", "waiting"):
            out[f"{key}_peak"] = max(series)
            out[f"{key}_mean"] = mean(series)
        elif key == "prefix_hit_rate" and "prefix_cache_hit_rate" not in out:
            out["prefix_cache_hit_rate"] = mean(series)
        elif key == "spec_accept_length" and "spec_mean_acceptance_length" not in out:
            out["spec_mean_acceptance_length"] = mean(series)
    return out


def describe_server(snapshot: Optional[Snapshot]) -> Dict[str, Any]:
    """Configuration that vLLM publishes through its metrics (cache_config_info etc.)."""
    if snapshot is None:
        return {}
    info: Dict[str, Any] = {}
    cache = snapshot.info.get("vllm:cache_config_info", {})
    if cache:
        for key in ("block_size", "num_gpu_blocks", "cache_dtype", "gpu_memory_utilization", "enable_prefix_caching"):
            if key in cache:
                info[key] = cache[key]
        try:
            info["kv_cache_capacity_tokens"] = int(float(cache["num_gpu_blocks"])) * int(float(cache["block_size"]))
        except (KeyError, ValueError):
            pass
        if "enable_prefix_caching" in cache:
            info["prefix_caching"] = cache["enable_prefix_caching"].lower() == "true"
        if "gpu_memory_utilization" in cache:
            try:
                info["gpu_memory_utilization"] = float(cache["gpu_memory_utilization"])
            except ValueError:
                pass
    info["speculative_decoding"] = snapshot.has_prefix("vllm:spec_decode") or "sglang:spec_accept_length" in snapshot.values
    return info


class ServerMetricsPoller:
    """Polls a metrics source in the background for gauges; snapshots counters on demand."""

    def __init__(self, fetch: Callable[[], Optional[str]], interval: float = 0.5, poll: bool = True):
        self.fetch = fetch
        self.interval = interval
        self.poll = poll
        self.gauge_samples: List[Dict[str, Any]] = []
        self.latest_gauges: Dict[str, Any] = {}
        self.available = self.snapshot() is not None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def snapshot(self) -> Optional[Snapshot]:
        try:
            text = self.fetch()
        except Exception:
            return None
        return parse_prometheus(text) if text else None

    def start(self) -> None:
        if not self.available or not self.poll or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="server-metrics")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            snap = self.snapshot()
            if snap is not None:
                gauges = {k: snap.gauge(k) for k in GAUGES}
                gauges["t"] = snap.t
                self.gauge_samples.append(gauges)
                self.latest_gauges = gauges
            time.sleep(max(0.0, self.interval - (time.perf_counter() - t0)))

    def gauges_between(self, t0: float, t1: float) -> List[Dict[str, Any]]:
        return [g for g in list(self.gauge_samples) if t0 <= g["t"] <= t1]


def http_metrics_fetcher(base_url: str, timeout: float = 5.0) -> Callable[[], Optional[str]]:
    url = f"{base_url}/metrics"

    def fetch() -> Optional[str]:
        r = requests.get(url, timeout=timeout)
        return r.text if r.status_code == 200 else None

    return fetch
