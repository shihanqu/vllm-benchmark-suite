"""Minimal stand-in for the vllm package so the offline engine can be tested without a GPU."""

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List

__version__ = "0.0.0+fake"

_STATE = {"prompt_tokens": 0.0, "generation_tokens": 0.0, "requests": 0.0}


class SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass
class Counter:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Gauge:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Histogram:
    name: str
    sum: float
    count: float
    labels: Dict[str, str] = field(default_factory=dict)


class LLM:
    last_kwargs = None

    def __init__(self, **kwargs):
        LLM.last_kwargs = kwargs
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=kwargs.get("max_model_len") or 8192))

    def generate(self, inputs, sampling_params=None, use_tqdm=True):
        n = len(inputs)
        prompt_len = len(inputs[0]["prompt_token_ids"])
        out_len = sampling_params.max_tokens
        # prefill cost + per-step decode cost growing with batch size
        time.sleep(prompt_len * n * 2e-7 + out_len * (0.0004 + 0.00002 * n))
        _STATE["prompt_tokens"] += prompt_len * n
        _STATE["generation_tokens"] += out_len * n
        _STATE["requests"] += n
        return [
            SimpleNamespace(
                prompt_token_ids=list(inp["prompt_token_ids"]),
                outputs=[SimpleNamespace(token_ids=list(range(out_len)))],
            )
            for inp in inputs
        ]

    def get_metrics(self) -> List[object]:
        return [
            Counter("vllm:prompt_tokens", _STATE["prompt_tokens"]),
            Counter("vllm:generation_tokens", _STATE["generation_tokens"]),
            Counter("vllm:prefix_cache_queries", _STATE["prompt_tokens"]),
            Counter("vllm:prefix_cache_hits", 0.0),
            Gauge("vllm:kv_cache_usage_perc", 0.0),
            Gauge("vllm:cache_config_info", 1.0, {"block_size": "16", "num_gpu_blocks": "4096", "enable_prefix_caching": "True"}),
            Histogram("vllm:e2e_request_latency_seconds", _STATE["requests"] * 0.1, _STATE["requests"]),
        ]
