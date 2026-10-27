#!/usr/bin/env python3
"""
Mock OpenAI-compatible inference server for exercising the v4 suite without a GPU.

Simulates a continuous-batching engine (chunked prefill, per-step decode cost that grows with
batch size and context, KV-cache capacity with queueing and preemption, optional speculative
decoding) and exposes vLLM-style /v1/models, /version, /tokenize and Prometheus /metrics.
If MOCK_GPU_STATE is set, the current load is written there for tests/v4/fake_nvidia_smi.py.

    python tests/v4/mock_server.py --port 8123 --speed 20 [--spec]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aiohttp import web


@dataclass
class Req:
    prompt_len: int
    max_tokens: int
    arrival: float
    queue: asyncio.Queue
    prefilled: int = 0
    generated: int = 0
    scheduled_at: Optional[float] = None
    first_token_at: Optional[float] = None


@dataclass
class Stats:
    counters: Dict[str, float] = field(default_factory=dict)
    hist: Dict[str, List[float]] = field(default_factory=dict)  # name -> [sum, count]

    def inc(self, name: str, v: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + v

    def observe(self, name: str, v: float) -> None:
        s = self.hist.setdefault(name, [0.0, 0.0])
        s[0] += v
        s[1] += 1


class Engine:
    def __init__(self, args):
        self.args = args
        self.speed = args.speed
        self.waiting: List[Req] = []
        self.running: List[Req] = []
        self.stats = Stats()
        self.kv_capacity = args.kv_capacity
        self.seen_prefixes = set()
        self.wakeup = asyncio.Event()

    def kv_used(self) -> int:
        return sum(r.prefilled + r.generated for r in self.running)

    async def loop(self) -> None:
        while True:
            if not self.running and not self.waiting:
                self.wakeup.clear()
                self._write_gpu_state()
                await self.wakeup.wait()
            now = time.perf_counter()
            # Admit waiting requests while sequences and KV capacity allow
            while self.waiting and len(self.running) < self.args.max_num_seqs:
                r = self.waiting[0]
                if self.kv_used() + r.prompt_len + 16 > self.kv_capacity:
                    break
                self.waiting.pop(0)
                r.scheduled_at = now
                self.stats.observe("vllm:request_queue_time_seconds", now - r.arrival)
                self.running.append(r)
            # Preempt the newest request if decode would overflow the KV cache
            while self.running and self.kv_used() + len(self.running) > self.kv_capacity and len(self.running) > 1:
                victim = self.running.pop()
                victim.prefilled = 0
                self.waiting.insert(0, victim)
                self.stats.inc("vllm:num_preemptions_total")

            # Chunked prefill with a per-step token budget
            budget = self.args.prefill_chunk
            prefill_tokens = 0
            for r in self.running:
                if r.prefilled < r.prompt_len and budget > 0:
                    take = min(budget, r.prompt_len - r.prefilled)
                    r.prefilled += take
                    budget -= take
                    prefill_tokens += take
            decoding = [r for r in self.running if r.prefilled >= r.prompt_len]
            ctx = sum(r.prefilled + r.generated for r in decoding)
            step = (
                self.args.step_base_ms
                + self.args.step_per_seq_ms * len(decoding)
                + self.args.step_per_kctx_ms * ctx / 1000.0
                + 1000.0 * prefill_tokens / self.args.prefill_tps
            ) / 1000.0
            await asyncio.sleep(step / self.speed)
            now = time.perf_counter()
            finished = []
            for r in decoding:
                k = 1
                if self.args.spec:
                    drafted = 3
                    accepted = sum(1 for _ in range(drafted) if random.random() < 0.55)
                    k = 1 + accepted
                    self.stats.inc("vllm:spec_decode_num_drafts_total")
                    self.stats.inc("vllm:spec_decode_num_draft_tokens_total", drafted)
                    self.stats.inc("vllm:spec_decode_num_accepted_tokens_total", accepted)
                k = min(k, r.max_tokens - r.generated)
                r.generated += k
                self.stats.inc("vllm:generation_tokens_total", k)
                if r.first_token_at is None:
                    r.first_token_at = now
                    self.stats.observe("vllm:time_to_first_token_seconds", now - r.arrival)
                    self.stats.observe("vllm:request_prefill_time_seconds", now - (r.scheduled_at or r.arrival))
                done = r.generated >= r.max_tokens
                r.queue.put_nowait((k, done))
                if done:
                    finished.append(r)
                    self.stats.observe("vllm:request_decode_time_seconds", now - r.first_token_at)
                    self.stats.observe("vllm:e2e_request_latency_seconds", now - r.arrival)
            for r in finished:
                self.running.remove(r)
            self._write_gpu_state()

    def _write_gpu_state(self) -> None:
        path = os.environ.get("MOCK_GPU_STATE")
        if not path:
            return
        load = min(1.0, len(self.running) / 16.0)
        state = {"running": len(self.running), "load": load, "kv": self.kv_used() / self.kv_capacity, "t": time.time()}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)

    def submit(self, prompt_len: int, max_tokens: int, prefix_key) -> Req:
        r = Req(prompt_len, max_tokens, time.perf_counter(), asyncio.Queue())
        self.stats.inc("vllm:prompt_tokens_total", prompt_len)
        self.stats.inc("vllm:prefix_cache_queries_total", prompt_len)
        if prefix_key in self.seen_prefixes:
            self.stats.inc("vllm:prefix_cache_hits_total", prompt_len)
        self.seen_prefixes.add(prefix_key)
        self.waiting.append(r)
        self.wakeup.set()
        return r

    def metrics_text(self) -> str:
        a = self.args
        lines = [
            "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig",
            "# TYPE vllm:cache_config_info gauge",
            f'vllm:cache_config_info{{block_size="16",cache_dtype="auto",enable_prefix_caching="True",'
            f'gpu_memory_utilization="0.9",num_gpu_blocks="{self.kv_capacity // 16}"}} 1.0',
            f'vllm:num_requests_running{{engine="0",model_name="{a.model}"}} {len(self.running)}',
            f'vllm:num_requests_waiting{{engine="0",model_name="{a.model}"}} {len(self.waiting)}',
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="{a.model}"}} {min(1.0, self.kv_used() / self.kv_capacity)}',
        ]
        for name, v in self.stats.counters.items():
            lines.append(f'{name}{{engine="0",model_name="{a.model}"}} {v}')
        for name, (s, c) in self.stats.hist.items():
            lines.append(f'{name}_bucket{{le="+Inf",model_name="{a.model}"}} {c}')
            lines.append(f'{name}_sum{{model_name="{a.model}"}} {s}')
            lines.append(f'{name}_count{{model_name="{a.model}"}} {c}')
        return "\n".join(lines) + "\n"


def build_app(args) -> web.Application:
    app = web.Application(client_max_size=1024**3)
    engine = Engine(args)

    async def on_startup(app):
        app["engine_task"] = asyncio.create_task(engine.loop())

    async def models(request):
        return web.json_response(
            {"object": "list", "data": [{"id": args.model, "object": "model", "owned_by": "vllm", "root": args.model,
                                         "max_model_len": args.max_model_len}]}
        )

    async def version(request):
        return web.json_response({"version": "0.11.2+mock"})

    async def tokenize(request):
        body = await request.json()
        n = len(str(body.get("prompt", "")).split())
        return web.json_response({"count": int(n * 1.31), "max_model_len": args.max_model_len})

    async def metrics(request):
        return web.Response(text=engine.metrics_text(), content_type="text/plain")

    async def completions(request, chat: bool = False):
        body = await request.json()
        if body.get("model") != args.model:
            return web.json_response({"error": {"message": f"model {body.get('model')} not found"}}, status=404)
        if chat:
            text = body["messages"][-1]["content"]
            prompt_len = int(len(text.split()) * 1.31) + 12
            key = text[:64]
        else:
            prompt = body["prompt"]
            if isinstance(prompt, list):
                prompt_len = len(prompt)
                key = tuple(prompt[:16])
            else:
                prompt_len = int(len(prompt.split()) * 1.31)
                key = prompt[:64]
        max_tokens = int(body.get("max_tokens", 16))
        if prompt_len + max_tokens > args.max_model_len:
            return web.json_response(
                {"error": {"message": f"maximum context length is {args.max_model_len} tokens"}}, status=400
            )
        if not body.get("ignore_eos") and random.random() < 0.3:
            max_tokens = max(1, max_tokens // 2)  # pretend the model hit EOS early
        r = engine.submit(prompt_len, max_tokens, key)

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        generated = 0
        while True:
            k, done = await r.queue.get()
            generated += k
            text = "tok " * k
            if chat:
                choice = {"index": 0, "delta": {"content": text}, "finish_reason": "length" if done else None}
                obj = {"object": "chat.completion.chunk", "choices": [choice]}
            else:
                choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": "length" if done else None}
                obj = {"object": "text_completion", "choices": [choice]}
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())
            if done:
                break
        usage = {"prompt_tokens": prompt_len, "completion_tokens": generated, "total_tokens": prompt_len + generated}
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def chat(request):
        return await completions(request, chat=True)

    app.on_startup.append(on_startup)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/version", version)
    app.router.add_get("/health", lambda r: web.Response(text=""))
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat)
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--model", default="mock/Qwen3-Mock-30B-A3B-NVFP4")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=48)
    p.add_argument("--kv-capacity", type=int, default=120000, help="KV cache capacity in tokens")
    p.add_argument("--prefill-tps", type=float, default=40000.0, help="Prefill tokens per second")
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--step-base-ms", type=float, default=7.0)
    p.add_argument("--step-per-seq-ms", type=float, default=0.18)
    p.add_argument("--step-per-kctx-ms", type=float, default=0.02)
    p.add_argument("--speed", type=float, default=1.0, help="Run the simulated clock N times faster")
    p.add_argument("--spec", action="store_true", help="Simulate speculative decoding (multi-token chunks)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    web.run_app(build_app(args), port=args.port, print=lambda *a: None)
