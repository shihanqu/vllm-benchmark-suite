"""Rich live dashboard (v3 layout: header, overall progress, current test, queue) plus recent results."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from rich import box
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engines import BatchProgress
from .util import fmt_duration, fmt_shape, fmt_tokens


def _work(test: Tuple[int, int, int], passes: int) -> float:
    """Rough relative cost of a test for the ETA (decode tokens dominate)."""
    i, o, b = test
    return passes * b * (i / 8.0 + o)


class LiveState:
    def __init__(self, queue: List[Tuple[int, int, int]], passes: int, paired: bool, gpu=None, poller=None):
        self.queue = list(queue)
        self.total = len(queue)
        self.passes = passes
        self.paired = paired
        self.gpu = gpu
        self.poller = poller
        self.progress = BatchProgress()
        self.test_num = 0
        self.current: Optional[Tuple[int, int, int]] = None
        self.phase = "starting"
        self.iteration = 0
        self.iterations_total = 0
        self.run_start = time.perf_counter()
        self.test_start = self.run_start
        self.recent: Deque[Dict[str, Any]] = deque(maxlen=6)
        self.work_done = 0.0
        self.time_done = 0.0
        self.label = ""

    # ----- updates from the runner -----------------------------------------------------
    def begin_test(self, index: int, test: Tuple[int, int, int]) -> None:
        self.test_num = index + 1
        self.current = test
        self.test_start = time.perf_counter()
        self.phase = "warmup"

    def set_phase(self, phase: str, iteration: int = 0, total: int = 0) -> None:
        self.phase = phase
        self.iteration = iteration
        self.iterations_total = total

    def end_test(self, row: Optional[Dict[str, Any]]) -> None:
        if self.current:
            self.work_done += _work(self.current, self.passes)
            self.time_done += time.perf_counter() - self.test_start
        if row is not None:
            self.recent.appendleft(row)
        self.phase = "cooldown"

    # ----- rendering -------------------------------------------------------------------
    def _eta(self) -> Optional[float]:
        if self.work_done <= 0:
            return None
        remaining = [t for t in self.queue[self.test_num :]]
        work_left = sum(_work(t, self.passes) for t in remaining)
        if self.current and self.phase != "cooldown":
            work_left += _work(self.current, self.passes) * 0.5
        return self.time_done / self.work_done * work_left

    def render(self) -> Layout:
        now = time.perf_counter()
        remaining = self.queue[self.test_num :]
        max_queue_items = 8
        queue_size = min(max(5, min(len(remaining), max_queue_items) + 4), 13)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="overall_progress", size=5),
            Layout(name="current_test", size=12),
            Layout(name="recent", size=10),
            Layout(name="remaining", size=queue_size),
        )

        header = Text()
        header.append("vLLM Benchmark v4 (LLM-Inference-Bench method) - ", style="bold cyan")
        header.append(f"Test {self.test_num}/{self.total}", style="bold yellow")
        header.append(f"  |  Runtime: {fmt_duration(now - self.run_start)}", style="dim")
        eta = self._eta()
        if eta is not None:
            header.append(f"  |  ETA ~{fmt_duration(eta)}", style="dim")
        if self.label:
            header.append(f"  |  {self.label}", style="dim")
        layout["header"].update(Panel(header, style="cyan"))

        done = max(0, self.test_num - (0 if self.phase == "cooldown" else 1))
        frac = done / self.total if self.total else 1.0
        bar_width = 60
        filled = int(frac * bar_width)
        progress_text = Text()
        progress_text.append("OVERALL PROGRESS\n", style="bold green")
        progress_text.append("█" * filled + "░" * (bar_width - filled) + " ", style="green")
        progress_text.append(f"{frac * 100:.1f}%\n", style="bold green")
        progress_text.append(f"Completed: {done}  Remaining: {self.total - done}", style="dim")
        layout["overall_progress"].update(Panel(progress_text, title="Benchmark Progress", border_style="green"))

        layout["current_test"].update(self._current_panel(now))
        layout["recent"].update(self._recent_panel())

        if remaining:
            queue_text = Text()
            queue_text.append(f"Queue ({len(remaining)} more tests):\n", style="bold blue")
            for i, o, b in remaining[:max_queue_items]:
                queue_text.append(f"  → {fmt_shape(i, o, self.paired)} tokens × batch {b}\n", style="dim")
            if len(remaining) > max_queue_items:
                queue_text.append(f"  ... and {len(remaining) - max_queue_items} more tests", style="italic grey50")
            layout["remaining"].update(Panel(queue_text, title="Upcoming Workload", border_style="blue"))
        else:
            layout["remaining"].update(Panel("Final test running", title="Queue", border_style="green"))
        return layout

    def _current_panel(self, now: float) -> Panel:
        info = Table(show_header=False, box=box.SIMPLE, border_style="cyan", padding=(0, 1))
        info.add_column("", style="cyan", width=14)
        info.add_column("", style="yellow")
        info.add_column("", style="cyan", width=12)
        info.add_column("", style="yellow")

        i, o, b = self.current or (0, 0, 0)
        phase = {
            "warmup": "[bold magenta]WARMUP[/bold magenta]",
            "probe": "[bold blue]TTFT PROBE[/bold blue]",
            "measure": "[bold yellow]MEASURING[/bold yellow]",
            "cooldown": "[green]COOLDOWN[/green]",
        }.get(self.phase, self.phase.upper())
        it = f"{self.iteration}/{self.iterations_total}" if self.iterations_total else "-"
        p = self.progress
        elapsed_batch = now - p.started
        rate = p.chunks / elapsed_batch if elapsed_batch > 0 and p.chunks else 0.0

        gpu = self.gpu.latest() if self.gpu and self.gpu.available else None
        gauges = self.poller.latest_gauges if self.poller and self.poller.available else {}

        def gpu_cell(key, fmt):
            return fmt.format(gpu[key]) if gpu and gpu.get(key) is not None else "-"

        util = gpu.get("gpu_util") if gpu else None
        util_color = "red" if (util or 0) > 95 else "yellow" if (util or 0) > 80 else "green"
        rows = [
            ("Input/Output", f"{fmt_tokens(i)} / {fmt_tokens(o)} tokens", "GPU util", f"[{util_color}]{util:.0f}%[/{util_color}]" if util is not None else "-"),
            ("Batch size", str(b), "Power", gpu_cell("power_draw", "{:.0f} W")),
            ("Phase", phase, "Temp", gpu_cell("temperature", "{:.0f} C")),
            ("Iteration", it, "SM clock", gpu_cell("gpu_clock", "{:.0f} MHz")),
            ("Requests", f"{p.done}/{p.size} done" + (f", [red]{p.failed} failed[/red]" if p.failed else ""), "KV cache", f"{gauges['kv_usage']:.0f}%" if gauges.get("kv_usage") is not None else "-"),
            ("Stream rate", f"{rate:,.0f} chunks/s" if rate else "-", "Run/Wait", f"{gauges.get('running') or 0:.0f} / {gauges.get('waiting') or 0:.0f}" if gauges else "-"),
            ("Test elapsed", fmt_duration(now - self.test_start), "", ""),
        ]
        for r in rows:
            info.add_row(*r)
        return Panel(info, title=f"Current Test ({self.test_num}/{self.total})", border_style="yellow")

    def _recent_panel(self) -> Panel:
        if not self.recent:
            return Panel(Text("No completed tests yet", style="dim"), title="Recent Results", border_style="magenta")
        t = Table(box=box.SIMPLE_HEAD, header_style="bold magenta", padding=(0, 1))
        for col in ("Shape", "Batch", "Gen tok/s", "Total tok/s", "Latency", "TTFT p50", "TPOT", "J/tok"):
            t.add_column(col, justify="right")
        for r in list(self.recent)[:5]:
            if r.get("status") == "failed":
                t.add_row(fmt_shape(r["input_len"], r["output_len"], self.paired), str(r["batch_size"]), "[red]FAILED[/red]", "", "", "", "", "")
                continue
            t.add_row(
                fmt_shape(r["input_len"], r["output_len"], self.paired),
                str(r["batch_size"]),
                f"{r.get('output_throughput', 0):,.0f}",
                f"{r.get('throughput', 0):,.0f}",
                f"{r.get('latency_avg', 0):.2f}s",
                f"{r['ttft_p50_ms']:.0f}ms" if r.get("ttft_p50_ms") is not None else "-",
                f"{r['tpot_mean_ms']:.1f}ms" if r.get("tpot_mean_ms") is not None else "-",
                f"{r['energy_per_output_token']:.3f}" if r.get("energy_per_output_token") is not None else "-",
            )
        return Panel(t, title="Recent Results", border_style="magenta")
