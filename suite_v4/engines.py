"""
Inference engines that execute one LLM-Inference-Bench batch at a time.

EndpointEngine  - any OpenAI-compatible server (vLLM, SGLang, ...). All batch requests are
                  released together and streamed, so TTFT and inter-token latency are measured
                  per request rather than estimated.
OfflineVLLMEngine - vLLM's in-process LLM() engine, exactly like LLM-Inference-Bench's
                  benchmark_latency.py: one llm.generate() call per batch, TTFT from a
                  max_tokens=1 probe batch (the paper's TTFT method).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import requests

from .config import BenchConfig

Prompt = Union[np.ndarray, List[int], str]


@dataclass
class RequestRecord:
    """Timing for one request. Times are seconds; `start` is a perf_counter() timestamp."""

    ok: bool = False
    start: float = 0.0
    ttft: Optional[float] = None
    e2e: Optional[float] = None
    itls: List[float] = field(default_factory=list)
    chunks: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    connection_error: bool = False


@dataclass
class BatchResult:
    start: float
    end: float
    records: List[RequestRecord]

    @property
    def latency(self) -> float:
        return self.end - self.start

    @property
    def ok_records(self) -> List[RequestRecord]:
        return [r for r in self.records if r.ok]


class BatchProgress:
    """Counters the live dashboard reads while a batch is in flight."""

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, size: int) -> None:
        self.size = size
        self.sent = 0
        self.done = 0
        self.failed = 0
        self.chunks = 0
        self.started = time.perf_counter()


class EngineError(RuntimeError):
    pass


def infer_quantization(*names: Optional[str]) -> Optional[str]:
    """Best-effort quantization label from model id/path (v3 missed NVFP4/MXFP4)."""
    text = " ".join(n for n in names if n)
    patterns = [
        (r"nvfp4", "NVFP4"),
        (r"mxfp4", "MXFP4"),
        (r"mxfp8", "MXFP8"),
        (r"fp8", "FP8"),
        (r"awq", "AWQ"),
        (r"gptq", "GPTQ"),
        (r"w4a16|int4|4bit|4-bit", "INT4"),
        (r"w8a8|int8|8bit|8-bit", "INT8"),
        (r"gguf|q4_k|q5_k|q8_0", "GGUF"),
        (r"bf16", "BF16"),
        (r"fp16", "FP16"),
    ]
    for pat, label in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return label
    return None


class EndpointEngine:
    kind = "endpoint"

    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.base = cfg.url
        self.model = cfg.model
        path = "/v1/completions" if cfg.api == "completions" else "/v1/chat/completions"
        self.request_url = f"{self.base}{path}"
        self.server_info: Dict[str, Any] = {}

    # ----- discovery -------------------------------------------------------------------
    def _get_json(self, path: str, timeout: float = 5.0) -> Optional[Any]:
        try:
            r = requests.get(f"{self.base}{path}", timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def discover(self) -> Dict[str, Any]:
        """Query /v1/models, /version and SGLang's server info; raise if the server is unreachable."""
        try:
            r = requests.get(f"{self.base}/v1/models", timeout=10)
        except Exception as e:
            raise EngineError(f"Cannot reach {self.base}: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"{self.base}/v1/models returned HTTP {r.status_code}")
        models = (r.json() or {}).get("data") or []
        entry = next((m for m in models if m.get("id") == self.model), models[0] if models else {})
        if not self.model:
            self.model = entry.get("id")
        if not self.model:
            raise EngineError("Server lists no models; pass --model")

        info: Dict[str, Any] = {
            "endpoint": self.base,
            "api": self.request_url.replace(self.base, ""),
            "model_name": self.model,
            "model_root": entry.get("root"),
            "max_model_len": entry.get("max_model_len"),
            "owned_by": entry.get("owned_by"),
        }
        version = self._get_json("/version")
        if isinstance(version, dict) and version.get("version"):
            info["version"] = version["version"]
        sglang = self._get_json("/get_server_info") or self._get_json("/server_info")
        if isinstance(sglang, dict) and ("tp_size" in sglang or "model_path" in sglang):
            info["framework"] = "SGLang"
            info["version"] = info.get("version") or sglang.get("version")
            info["tensor_parallel"] = sglang.get("tp_size")
            info["pipeline_parallel"] = sglang.get("pp_size")
            info["data_parallel"] = sglang.get("dp_size")
            info["quantization"] = sglang.get("quantization")
            info["max_num_seqs"] = sglang.get("max_running_requests")
            info["prefix_caching"] = not sglang.get("disable_radix_cache", False)
            if sglang.get("speculative_algorithm"):
                info["speculative_algorithm"] = sglang["speculative_algorithm"]
        elif (entry.get("owned_by") or "").lower() == "vllm" or info.get("version"):
            info["framework"] = "vLLM"
        else:
            info["framework"] = "OpenAI-compatible"
        info["quantization"] = info.get("quantization") or infer_quantization(self.model, entry.get("root"))
        self.server_info = info
        return info

    def metrics_fetcher(self) -> Callable[[], Optional[str]]:
        from .servermetrics import http_metrics_fetcher

        return http_metrics_fetcher(self.base)

    def alive(self) -> bool:
        return self._get_json("/v1/models", timeout=10) is not None

    def calibrate_tokens_per_word(self, text: str, n_words: int) -> Optional[float]:
        """Use vLLM's /tokenize endpoint to size text prompts; None if unsupported."""
        try:
            r = requests.post(f"{self.base}/tokenize", json={"model": self.model, "prompt": text}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                count = data.get("count") or len(data.get("tokens") or [])
                if count:
                    return count / n_words
        except Exception:
            pass
        return None

    def preflight(self, prompt: Prompt) -> Dict[str, Any]:
        """One tiny streamed request: proves the prompt format works and checks ignore_eos is honoured."""
        result = self.run_batch([prompt], output_len=16, progress=BatchProgress())
        rec = result.records[0]
        if not rec.ok:
            raise EngineError(f"Preflight request failed: {rec.error}")
        return {
            "ttft_ms": rec.ttft * 1000 if rec.ttft else None,
            "completion_tokens": rec.completion_tokens,
            "ignore_eos_honoured": rec.completion_tokens == 16,
            "usage_reported": rec.prompt_tokens is not None,
        }

    # ----- execution -------------------------------------------------------------------
    def _payload(self, prompt: Prompt, output_len: int) -> bytes:
        cfg = self.cfg
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": output_len,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if cfg.ignore_eos:
            body["ignore_eos"] = True
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if self.cfg.api == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        return json.dumps(body).encode()

    def run_batch(self, prompts: Sequence[Prompt], output_len: int, progress: BatchProgress) -> BatchResult:
        # Encode every payload before the clock starts so long prompts are released together.
        payloads = [self._payload(p, output_len) for p in prompts]
        return asyncio.run(self._run_batch(payloads, progress))

    async def _run_batch(self, payloads: List[bytes], progress: BatchProgress) -> BatchResult:
        import aiohttp

        progress.reset(len(payloads))
        timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            start = time.perf_counter()
            records = await asyncio.gather(*(self._stream_one(session, p, progress) for p in payloads))
            end = time.perf_counter()
        return BatchResult(start=start, end=end, records=list(records))

    def _is_token_chunk(self, choice: Dict[str, Any]) -> bool:
        if self.cfg.api == "chat":
            delta = choice.get("delta") or {}
            return bool(
                delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls")
            )
        text = choice.get("text")
        if text:
            return True
        # Empty text still marks a generated token (e.g. a partial UTF-8 sequence) unless the
        # chunk only carries the finish reason.
        return text == "" and choice.get("finish_reason") is None

    async def _stream_one(self, session, payload: bytes, progress: BatchProgress) -> RequestRecord:
        import aiohttp

        rec = RequestRecord()
        last = None
        rec.start = time.perf_counter()
        progress.sent += 1
        try:
            async with session.post(
                self.request_url, data=payload, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    rec.error = f"HTTP {resp.status}: {body.strip()[:300]}"
                else:
                    async for raw in resp.content:
                        line = raw.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        if obj.get("error"):
                            rec.error = str(obj["error"])[:300]
                            break
                        now = time.perf_counter()
                        choices = obj.get("choices") or []
                        if choices and self._is_token_chunk(choices[0]):
                            if rec.ttft is None:
                                rec.ttft = now - rec.start
                            else:
                                rec.itls.append(now - last)
                            last = now
                            rec.chunks += 1
                            progress.chunks += 1
                        usage = obj.get("usage")
                        if usage:
                            rec.prompt_tokens = usage.get("prompt_tokens")
                            rec.completion_tokens = usage.get("completion_tokens")
        except (aiohttp.ClientConnectorError, ConnectionRefusedError, aiohttp.ServerDisconnectedError) as e:
            rec.error = f"Connection error: {e}"
            rec.connection_error = True
        except asyncio.TimeoutError:
            rec.error = f"Timed out after {self.cfg.request_timeout:.0f}s"
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:300]
        rec.e2e = time.perf_counter() - rec.start
        rec.ok = rec.error is None and rec.ttft is not None
        if rec.ok and rec.completion_tokens is None:
            rec.completion_tokens = rec.chunks  # server did not report usage
        if not rec.ok and rec.error is None:
            rec.error = "Stream ended without tokens"
        progress.done += 1
        if not rec.ok:
            progress.failed += 1
        return rec

    def close(self) -> None:
        pass


class OfflineVLLMEngine:
    """vLLM LLM() in-process, as in LLM-Inference-Bench's benchmark_latency.py."""

    kind = "offline"

    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        import vllm
        from vllm import LLM, SamplingParams

        self._vllm = vllm
        self._SamplingParams = SamplingParams
        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "tensor_parallel_size": cfg.tensor_parallel or 1,
            "dtype": cfg.dtype,
            "trust_remote_code": cfg.trust_remote_code,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "seed": cfg.seed,
            "disable_log_stats": False,  # keeps llm.get_metrics() populated
        }
        if cfg.pipeline_parallel:
            kwargs["pipeline_parallel_size"] = cfg.pipeline_parallel
        if cfg.max_model_len:
            kwargs["max_model_len"] = cfg.max_model_len
        if cfg.enable_prefix_caching is not None:
            kwargs["enable_prefix_caching"] = cfg.enable_prefix_caching
        kwargs.update(cfg.engine_args)
        self.model = cfg.model
        self.llm = LLM(**kwargs)
        self.server_info: Dict[str, Any] = {}

    def discover(self) -> Dict[str, Any]:
        max_len = None
        try:
            max_len = self.llm.llm_engine.model_config.max_model_len
        except Exception:
            pass
        self.server_info = {
            "framework": "vLLM (offline LLM())",
            "version": getattr(self._vllm, "__version__", None),
            "model_name": self.model,
            "max_model_len": max_len,
            "tensor_parallel": self.cfg.tensor_parallel or 1,
            "pipeline_parallel": self.cfg.pipeline_parallel,
            "quantization": self.cfg.engine_args.get("quantization") or infer_quantization(self.model),
            "endpoint": "in-process",
            "api": "LLM.generate",
        }
        return self.server_info

    def metrics_fetcher(self) -> Optional[Callable[[], Optional[str]]]:
        """Render llm.get_metrics() (vLLM V1) as Prometheus text so the same parser applies."""
        if not hasattr(self.llm, "get_metrics"):
            return None

        def fmt_labels(labels: Optional[Dict[str, str]]) -> str:
            if not labels:
                return ""
            return "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"

        def fetch() -> Optional[str]:
            lines = []
            for m in self.llm.get_metrics():
                kind = type(m).__name__
                lab = fmt_labels(getattr(m, "labels", None))
                if kind in ("Counter", "Gauge"):
                    lines.append(f"{m.name}{lab} {m.value}")
                elif kind == "Histogram":
                    lines.append(f"{m.name}_sum{lab} {m.sum}")
                    lines.append(f"{m.name}_count{lab} {m.count}")
            return "\n".join(lines)

        return fetch

    def run_batch(self, prompts: Sequence[Prompt], output_len: int, progress: BatchProgress) -> BatchResult:
        cfg = self.cfg
        progress.reset(len(prompts))
        params = self._SamplingParams(
            n=1, temperature=cfg.temperature, top_p=cfg.top_p, ignore_eos=cfg.ignore_eos, max_tokens=output_len
        )
        inputs = [{"prompt_token_ids": p.tolist() if isinstance(p, np.ndarray) else list(p)} for p in prompts]
        progress.sent = len(inputs)
        start = time.perf_counter()
        outputs = self.llm.generate(inputs, sampling_params=params, use_tqdm=False)
        end = time.perf_counter()
        records = []
        for out in outputs:
            completion = len(out.outputs[0].token_ids) if out.outputs else 0
            records.append(
                RequestRecord(
                    ok=True,
                    start=start,
                    e2e=end - start,
                    prompt_tokens=len(out.prompt_token_ids or []),
                    completion_tokens=completion,
                )
            )
        progress.done = len(records)
        return BatchResult(start=start, end=end, records=records)

    def preflight(self, prompt: Prompt) -> Dict[str, Any]:
        result = self.run_batch([prompt], output_len=16, progress=BatchProgress())
        rec = result.records[0]
        return {"completion_tokens": rec.completion_tokens, "ignore_eos_honoured": rec.completion_tokens == 16}

    def calibrate_tokens_per_word(self, text: str, n_words: int) -> Optional[float]:
        return None

    def alive(self) -> bool:
        return True

    def close(self) -> None:
        pass
