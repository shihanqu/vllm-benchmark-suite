"""
Sweep execution: for every (input_len, output_len, batch_size) test run the LLM-Inference-Bench
loop - warmup batches, then measured batches - while sampling GPU and server telemetry over the
measured window only.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console

from .config import BenchConfig
from .dashboard import LiveState
from .engines import BatchResult
from .metrics import add_scaling_efficiency, aggregate_test
from .servermetrics import ServerMetricsPoller, derive_window
from .telemetry import GPUTelemetry
from .util import fmt_shape
from .workload import PHASE_MEASURE, PHASE_PROBE, PHASE_WARMUP, PromptFactory


class ServerUnavailable(RuntimeError):
    pass


class SweepRunner:
    def __init__(
        self,
        cfg: BenchConfig,
        engine,
        factory: PromptFactory,
        state: LiveState,
        console: Console,
        gpu: Optional[GPUTelemetry] = None,
        poller: Optional[ServerMetricsPoller] = None,
        max_model_len: Optional[int] = None,
        idle_power: Optional[float] = None,
        on_checkpoint: Optional[Callable[["SweepRunner"], None]] = None,
    ):
        self.cfg = cfg
        self.engine = engine
        self.factory = factory
        self.state = state
        self.console = console
        self.gpu = gpu if gpu and gpu.available else None
        self.poller = poller if poller and poller.available else None
        self.max_model_len = max_model_len
        self.idle_power = idle_power
        self.on_checkpoint = on_checkpoint
        self.results: List[Dict[str, Any]] = []
        self.request_rows: List[Dict[str, Any]] = []
        self.windows: List[Dict[str, Any]] = []
        self.origin = time.perf_counter()
        self.aborted: Optional[str] = None

    # ----- helpers ---------------------------------------------------------------------
    def _prompts(self, i: int, o: int, b: int, phase: int, k: int):
        if self.cfg.prompt_format == "token_ids":
            return self.factory.token_prompts(i, o, b, phase, k)
        return self.factory.text_prompts(i, o, b, phase, k)

    def _batch(self, i: int, o: int, b: int, phase: int, k: int, output_len: Optional[int] = None) -> BatchResult:
        prompts = self._prompts(i, o, b, phase, k)
        result = self.engine.run_batch(prompts, output_len or o, self.state.progress)
        if not result.ok_records and (any(r.connection_error for r in result.records) or not self.engine.alive()):
            raise ServerUnavailable(result.records[0].error if result.records else "no response")
        return result

    @staticmethod
    def _base_row(i: int, o: int, b: int, status: str, error: Optional[str]) -> Dict[str, Any]:
        row = {
            "input_len": i,
            "output_len": o,
            "batch_size": b,
            "status": status,
            "context_length": i,
            "concurrent_users": b,
        }
        if error:
            row["error_sample"] = error
        return row

    def _request_rows(self, test_index: int, i: int, o: int, b: int, k: int, result: BatchResult) -> None:
        for n, r in enumerate(result.records):
            tpot = None
            if r.ok and r.ttft is not None and (r.completion_tokens or 0) > 1:
                tpot = (r.e2e - r.ttft) / (r.completion_tokens - 1) * 1000
            self.request_rows.append(
                {
                    "test": test_index + 1,
                    "input_len": i,
                    "output_len": o,
                    "batch_size": b,
                    "iteration": k + 1,
                    "request": n,
                    "ok": r.ok,
                    "send_offset_ms": (r.start - result.start) * 1000,
                    "ttft_ms": r.ttft * 1000 if r.ttft is not None else None,
                    "tpot_ms": tpot,
                    "itl_max_ms": max(r.itls) * 1000 if r.itls else None,
                    "e2e_s": r.e2e,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "chunks": r.chunks,
                    "error": r.error,
                }
            )

    # ----- main loop -------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        queue = self.cfg.test_queue()
        for idx, (i, o, b) in enumerate(queue):
            self.state.begin_test(idx, (i, o, b))
            if self.cfg.skip_exceeding_context and self.max_model_len and i + o > self.max_model_len:
                row = self._base_row(i, o, b, "skipped", f"input+output {i + o} exceeds max_model_len {self.max_model_len}")
                self.results.append(row)
                self.console.print(
                    f"[yellow]SKIP[/yellow] Test {idx + 1}/{len(queue)} - {fmt_shape(i, o, self.cfg.paired_io)} × batch {b}: {row['error_sample']}"
                )
                self.state.end_test(row)
                continue
            try:
                row = self._run_test(idx, i, o, b)
            except ServerUnavailable as e:
                self.aborted = f"Server unavailable during {fmt_shape(i, o, self.cfg.paired_io)} × batch {b}: {e}"
                row = self._base_row(i, o, b, "failed", str(e))
                self.results.append(row)
                self.state.end_test(row)
                self.console.print(f"[bold red]ABORT[/bold red] {self.aborted}")
                break
            self.results.append(row)
            add_scaling_efficiency(self.results)
            self._print_line(idx, len(queue), row)
            self.state.end_test(row)
            if self.on_checkpoint:
                self.on_checkpoint(self)
            if idx < len(queue) - 1 and self.cfg.cooldown_seconds > 0:
                time.sleep(self.cfg.cooldown_seconds)
        add_scaling_efficiency(self.results)
        return self.results

    def _run_test(self, idx: int, i: int, o: int, b: int) -> Dict[str, Any]:
        cfg = self.cfg
        test_start = time.perf_counter()

        for w in range(cfg.warmup_iterations):
            self.state.set_phase("warmup", w + 1, cfg.warmup_iterations)
            result = self._batch(i, o, b, PHASE_WARMUP, w)
            if not result.ok_records:
                return self._base_row(i, o, b, "failed", f"warmup failed: {result.records[0].error}")

        probe_latency = None
        if self.engine.kind == "offline":
            # LLM-Inference-Bench TTFT: latency of the same batch generating a single token
            self.state.set_phase("probe", 1, 1)
            probe = self._batch(i, o, b, PHASE_PROBE, 0, output_len=1)
            probe_latency = probe.latency if probe.ok_records else None

        before = self.poller.snapshot() if self.poller else None
        energy0 = self.gpu.energy_joules() if self.gpu else None
        t0 = time.perf_counter()
        iterations: List[BatchResult] = []
        for k in range(cfg.iterations):
            self.state.set_phase("measure", k + 1, cfg.iterations)
            result = self._batch(i, o, b, PHASE_MEASURE, k)
            iterations.append(result)
            self._request_rows(idx, i, o, b, k, result)
            if not result.ok_records:
                break  # e.g. OOM / HTTP 500: further iterations would fail the same way
            if cfg.max_seconds_per_config and time.perf_counter() - t0 > cfg.max_seconds_per_config:
                break
        t1 = time.perf_counter()
        energy1 = self.gpu.energy_joules() if self.gpu else None
        after = self.poller.snapshot() if self.poller else None

        gpu_stats = self.gpu.window_stats(t0, t1) if self.gpu else None
        server_stats = derive_window(before, after, self.poller.gauges_between(t0, t1)) if self.poller else None
        energy = energy1 - energy0 if energy0 is not None and energy1 is not None else None
        self.windows.append(
            {
                "input_len": i,
                "output_len": o,
                "batch_size": b,
                "test_start": test_start - self.origin,
                "measure_start": t0 - self.origin,
                "measure_end": t1 - self.origin,
            }
        )
        row = aggregate_test(
            i,
            o,
            b,
            iterations,
            gpu_stats=gpu_stats,
            energy_counter_j=energy,
            window_seconds=t1 - t0,
            server_stats=server_stats,
            probe_latency=probe_latency,
            idle_power=self.idle_power,
        )
        return row

    def _print_line(self, idx: int, total: int, row: Dict[str, Any]) -> None:
        shape = fmt_shape(row["input_len"], row["output_len"], self.cfg.paired_io)
        head = f"Test {idx + 1}/{total} - {shape} × batch {row['batch_size']}"
        if row.get("status") == "failed":
            self.console.print(f"[red]FAIL[/red] {head}: {row.get('error_sample', 'all requests failed')}")
            return
        tag = "[green]OK[/green]" if row["status"] == "ok" else "[yellow]PARTIAL[/yellow]"
        parts = [
            f"[bold yellow]{row['output_throughput']:,.1f}[/bold yellow] gen tok/s",
            f"{row['throughput']:,.0f} total tok/s",
            f"[bold cyan]{row['latency_avg']:.2f}s[/bold cyan] batch latency",
        ]
        if row.get("ttft_p50_ms") is not None:
            parts.append(f"TTFT p50 {row['ttft_p50_ms']:.0f}ms")
        if row.get("tpot_mean_ms") is not None:
            parts.append(f"TPOT {row['tpot_mean_ms']:.1f}ms")
        if row.get("avg_gpu_util") is not None:
            parts.append(f"[bold magenta]{row['avg_gpu_util']:.0f}%[/bold magenta] GPU")
        if row.get("energy_per_output_token") is not None:
            parts.append(f"{row['energy_per_output_token']:.3f} J/tok")
        if row["status"] == "partial":
            parts.append(f"[yellow]{row['success_rate']:.0f}% ok[/yellow]")
        fidelity = row.get("output_token_fidelity")
        if fidelity is not None and abs(fidelity - 100.0) > 0.5:
            parts.append(f"[yellow]output {fidelity:.0f}% of requested[/yellow]")
        self.console.print(f"{tag} {head}: " + ", ".join(parts))

    def timeline(self) -> Dict[str, Any]:
        return self.gpu.timeline(self.origin) if self.gpu else {}

    def remaining_tests(self) -> List[Tuple[int, int, int]]:
        done = {(r["input_len"], r["output_len"], r["batch_size"]) for r in self.results}
        return [t for t in self.cfg.test_queue() if t not in done]
