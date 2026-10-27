"""
Command line entry point.

    python vllm_benchmark_suitev4.py                         # interactive, like v3
    python vllm_benchmark_suitev4.py run --config configs/v4/example.toml
    python vllm_benchmark_suitev4.py run --url http://localhost:1236 --preset quick -y
    python vllm_benchmark_suitev4.py run --engine offline --model meta-llama/Meta-Llama-3-8B --tp 1
    python vllm_benchmark_suitev4.py report outputs/benchmark_v4_<label>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from . import __version__
from .config import (
    PRESET_DESCRIPTIONS,
    PRESETS,
    BenchConfig,
    build_config,
    load_config_file,
    parse_int_list,
)
from .dashboard import LiveState
from .engines import EndpointEngine, EngineError, OfflineVLLMEngine
from .output import output_paths, write_argonne_csv, write_json, write_requests_csv, write_results_csv
from .runner import SweepRunner
from .servermetrics import ServerMetricsPoller, describe_server
from .telemetry import GPUTelemetry, get_system_info
from .util import fmt_duration, fmt_tokens, sanitize_filename
from .workload import PHASE_PROBE, PromptFactory

console = Console()

METHODOLOGY = {
    "name": "LLM-Inference-Bench (Chitty-Venkata et al., Argonne National Laboratory, arXiv:2411.00136)",
    "workload": "Batches of B requests released together; random token-id prompts of exactly input_len tokens; "
    "ignore_eos so every request generates exactly output_len tokens; fresh prompts per iteration.",
    "latency": "Mean wall time for the whole batch to finish, over measured iterations (warmup excluded).",
    "throughput": "B * (input_len + output_len) / latency (total tokens per second).",
    "output_throughput": "Generated tokens per second (v3 'tokens/sec').",
    "ttft_tpot_itl": "Measured per request from the token stream (endpoint) or a max_tokens=1 probe batch (offline).",
    "energy": "GPU hardware energy counter over the measured window (NVML), else integrated power samples.",
}


# ----------------------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------------------
def _add_run_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("workload (LLM-Inference-Bench grid)")
    g.add_argument("--config", help="TOML or JSON config file (see configs/v4/)")
    g.add_argument("--preset", choices=list(PRESETS), help="Workload preset (default: standard)")
    g.add_argument("--batch-sizes", help="Comma list, e.g. 1,16,32,64")
    g.add_argument("--io-lengths", help="Input = output lengths, e.g. 128,256,512,1024,2048")
    g.add_argument("--input-lengths", help="Input lengths (cross product with --output-lengths), e.g. 1K,4K,16K")
    g.add_argument("--output-lengths", help="Output lengths, e.g. 500")
    g.add_argument("--warmup", type=int, dest="warmup_iterations", help="Warmup batches per test")
    g.add_argument("--iterations", type=int, help="Measured batches per test")
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float, dest="top_p")
    g.add_argument("--seed", type=int)
    g.add_argument("--cooldown", type=float, dest="cooldown_seconds", help="Pause between tests (s)")
    g.add_argument("--timeout", type=float, dest="request_timeout", help="Per-request timeout (s)")
    g.add_argument("--max-seconds-per-config", type=float, help="Stop measuring a test after this many seconds")
    g.add_argument("--no-skip-context", action="store_true", help="Run shapes longer than max_model_len anyway")

    e = p.add_argument_group("engine")
    e.add_argument("--engine", choices=["endpoint", "offline"])
    e.add_argument("--url", help="Server base URL (default http://localhost:8000)")
    e.add_argument("--api", choices=["completions", "chat"])
    e.add_argument("--model", help="Served model id (endpoint) or model path (offline)")
    e.add_argument("--prompt-format", choices=["token_ids", "text"])
    e.add_argument("--tp", type=int, dest="tensor_parallel", help="Tensor parallel size (offline engine / metadata)")
    e.add_argument("--pp", type=int, dest="pipeline_parallel", help="Pipeline parallel size (offline engine / metadata)")
    e.add_argument("--dtype", help="Offline engine dtype (LLM-Inference-Bench used float16)")
    e.add_argument("--max-model-len", type=int)
    e.add_argument("--gpu-memory-utilization", type=float)
    e.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE", help="Extra vLLM LLM() argument (repeatable)")

    m = p.add_argument_group("labels and telemetry")
    m.add_argument("--label", help="Display name for titles and file names (e.g. qwen3.6-35b-nvfp4-300W)")
    m.add_argument("--hardware", help="Override the detected GPU description")
    m.add_argument("--framework", help="Override the detected framework name")
    m.add_argument("--quantization", help="Override the detected quantization label")
    m.add_argument("--num-gpus", type=int, help="'Num of Hardware' for the LLM-Inference-Bench CSV")
    m.add_argument("--gpu-indices", help="GPUs to monitor (nvidia-smi indices), e.g. 0,1")
    m.add_argument("--gpu-poll-interval", type=float)
    m.add_argument("--no-gpu-monitor", action="store_true")
    m.add_argument("--no-server-metrics", action="store_true")
    m.add_argument("--output-dir")
    m.add_argument("--dpi", type=int, dest="png_dpi")

    u = p.add_argument_group("interface")
    u.add_argument("-i", "--interactive", action="store_true", help="Prompt for settings like v3")
    u.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
    u.add_argument("--no-live", action="store_true", help="Plain log lines instead of the live dashboard")
    u.add_argument("--details", action="store_true", help="Print the detailed per-shape summary at the end")


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    for key in (
        "preset", "warmup_iterations", "iterations", "temperature", "top_p", "seed", "cooldown_seconds",
        "request_timeout", "max_seconds_per_config", "engine", "url", "api", "model", "prompt_format",
        "tensor_parallel", "pipeline_parallel", "dtype", "max_model_len", "gpu_memory_utilization", "label",
        "hardware", "framework", "quantization", "num_gpus", "gpu_poll_interval", "output_dir", "png_dpi",
    ):
        value = getattr(args, key, None)
        if value is not None:
            o[key] = value
    for key in ("batch_sizes", "io_lengths", "input_lengths", "output_lengths", "gpu_indices"):
        value = parse_int_list(getattr(args, key, None))
        if value:
            o[key] = value
    if args.no_skip_context:
        o["skip_exceeding_context"] = False
    if args.no_gpu_monitor:
        o["gpu_monitor"] = False
    if args.no_server_metrics:
        o["server_metrics"] = False
    if args.engine_arg:
        engine_args = {}
        for item in args.engine_arg:
            key, _, raw = item.partition("=")
            try:
                engine_args[key.strip()] = json.loads(raw)
            except ValueError:
                engine_args[key.strip()] = raw
        o["engine_args"] = engine_args
    return o


# ----------------------------------------------------------------------------------------
# Interactive configuration (v3 style)
# ----------------------------------------------------------------------------------------
def interactive_config(cfg: BenchConfig, file_values: Dict[str, Any], overrides: Dict[str, Any]) -> BenchConfig:
    console.print("\n[bold cyan]••• Benchmark Configuration •••[/bold cyan]\n")
    if cfg.engine == "endpoint":
        overrides["url"] = Prompt.ask("Server URL", default=cfg.url)

    names = list(PRESET_DESCRIPTIONS)
    console.print("[bold]Select workload preset:[/bold]")
    for n, name in enumerate(names, 1):
        console.print(f"  [{n}] {name:<13} {PRESET_DESCRIPTIONS[name]}")
    console.print(f"  [{len(names) + 1}] custom        Enter batch sizes and lengths\n")
    choice = IntPrompt.ask("Select preset", default=1, choices=[str(i) for i in range(1, len(names) + 2)])
    if choice <= len(names):
        overrides["preset"] = names[choice - 1]
        for key in ("batch_sizes", "io_lengths", "input_lengths", "output_lengths", "warmup_iterations", "iterations"):
            overrides.pop(key, None)
            file_values.pop(key, None)
    else:
        overrides["batch_sizes"] = parse_int_list(Prompt.ask("Batch sizes", default="1,16,32,64"))
        if Confirm.ask("Use input = output lengths (LLM-Inference-Bench style)?", default=True):
            overrides["io_lengths"] = parse_int_list(Prompt.ask("Input/output lengths", default="128,256,512,1024,2048"))
        else:
            overrides["input_lengths"] = parse_int_list(Prompt.ask("Input lengths", default="1K,4K,16K"))
            overrides["output_lengths"] = parse_int_list(Prompt.ask("Output lengths", default="500"))
        overrides["warmup_iterations"] = IntPrompt.ask("Warmup batches per test", default=2)
        overrides["iterations"] = IntPrompt.ask("Measured batches per test", default=5)

    label = Prompt.ask("Label for titles/file names (Enter = model id)", default=cfg.label or "")
    if label:
        overrides["label"] = label
    return build_config(file_values=file_values, overrides=overrides)


def print_config_summary(cfg: BenchConfig) -> None:
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Preset", cfg.preset)
    if cfg.paired_io:
        table.add_row("Input = Output", ", ".join(fmt_tokens(n) for n in cfg.io_lengths))
    else:
        table.add_row("Input Lengths", ", ".join(fmt_tokens(n) for n in cfg.input_lengths))
        table.add_row("Output Lengths", ", ".join(fmt_tokens(n) for n in cfg.output_lengths))
    table.add_row("Batch Sizes", ", ".join(str(b) for b in cfg.batch_sizes))
    table.add_row("Warmup / Measured", f"{cfg.warmup_iterations} / {cfg.iterations} batches per test")
    table.add_row("Sampling", f"temperature={cfg.temperature}, top_p={cfg.top_p}, ignore_eos={cfg.ignore_eos}")
    table.add_row("Prompts", "random token ids" if cfg.prompt_format == "token_ids" else "random words (text)")
    table.add_row("Seed", str(cfg.seed))
    table.add_row("Total Tests", str(len(cfg.test_queue())))
    console.print(table)


# ----------------------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------------------
def print_system_panels(system_info: Dict[str, Any], server_info: Dict[str, Any], gpu: GPUTelemetry) -> None:
    sys_table = Table(show_header=False, box=box.ROUNDED, border_style="green")
    sys_table.add_column("", style="cyan bold")
    sys_table.add_column("", style="yellow")
    sys_table.add_row("Python", system_info["python_version"])
    sys_table.add_row("Platform", system_info["platform"])
    if system_info.get("gpu_name"):
        vram = system_info.get("total_vram_gb")
        sys_table.add_row("GPU", system_info["gpu_name"] + (f" ({vram:.0f}GB)" if vram else ""))
    if system_info.get("gpu_description"):
        sys_table.add_row("GPU (label)", system_info["gpu_description"])
    limits = [x for x in system_info.get("power_limits_w") or [] if x]
    if limits:
        sys_table.add_row("Power Limit", " / ".join(f"{x:.0f}W" for x in limits))
    if system_info.get("driver_version"):
        sys_table.add_row("Driver", system_info["driver_version"])
    if system_info.get("cuda_version"):
        sys_table.add_row("CUDA", system_info["cuda_version"])
    sys_table.add_row("Telemetry", gpu.backend_name or f"[red]unavailable[/red] ({gpu.error})")
    console.print(Panel(sys_table, title="[bold green]System Information[/bold green]", border_style="green"))

    srv = Table(show_header=False, box=box.ROUNDED, border_style="blue")
    srv.add_column("", style="cyan bold")
    srv.add_column("", style="yellow")
    rows = [
        ("Model", server_info.get("model_name")),
        ("Endpoint", f"{server_info.get('endpoint')} {server_info.get('api') or ''}"),
        ("Framework", " ".join(str(x) for x in (server_info.get("framework"), server_info.get("version")) if x)),
        ("Quantization", server_info.get("quantization")),
        ("Tensor Parallel", server_info.get("tensor_parallel")),
        ("Pipeline Parallel", server_info.get("pipeline_parallel")),
        ("Max Context", f"{server_info['max_model_len']:,} tokens" if server_info.get("max_model_len") else None),
        ("KV Cache Capacity", f"{server_info['kv_cache_capacity_tokens']:,} tokens" if server_info.get("kv_cache_capacity_tokens") else None),
        ("Prefix Caching", ("Enabled" if server_info["prefix_caching"] else "Disabled") if server_info.get("prefix_caching") is not None else None),
        ("GPU Mem Util", f"{server_info['gpu_memory_utilization']:.0%}" if isinstance(server_info.get("gpu_memory_utilization"), float) else None),
        ("Spec Decoding", "detected" if server_info.get("speculative_decoding") else None),
        ("Server Metrics", "available" if server_info.get("metrics_available") else "unavailable"),
    ]
    for k, v in rows:
        if v not in (None, ""):
            srv.add_row(k, str(v))
    console.print(Panel(srv, title="[bold blue]Inference Server[/bold blue]", border_style="blue"))


# ----------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------
def _build_package(cfg, runner, system_info, server_info, telemetry_meta, preflight, start_wall, complete) -> Dict[str, Any]:
    configuration = cfg.to_dict()
    # v3-compatible keys (consolidate_data.py reads these)
    configuration["context_lengths"] = sorted({i for i, _ in cfg.shapes()})
    configuration["concurrent_users"] = list(cfg.batch_sizes)
    configuration["output_tokens"] = sorted({o for _, o in cfg.shapes()})[0] if cfg.shapes() else None
    return {
        "metadata": {
            "suite_version": __version__,
            "methodology": METHODOLOGY,
            "complete": complete,
            "aborted": runner.aborted,
            "timestamp": start_wall.strftime("%Y%m%d_%H%M%S"),
            "started_at": start_wall.isoformat(),
            "benchmark_duration": time.perf_counter() - runner.origin,
            "label": cfg.label,
            "model": server_info.get("model_name"),
            "system_info": system_info,
            "server_info": server_info,
            "configuration": configuration,
            "telemetry": telemetry_meta,
            "preflight": preflight,
            "windows": runner.windows,
            "timeline": runner.timeline(),
        },
        "results": runner.results,
    }


def cmd_run(args: argparse.Namespace) -> int:
    file_values = load_config_file(args.config) if args.config else {}
    overrides = _overrides_from_args(args)
    cfg = build_config(file_values=file_values, overrides=overrides)

    console.print(
        Panel.fit(
            "[bold cyan]vLLM Performance Benchmark Suite v4[/bold cyan]\n"
            "[dim]LLM-Inference-Bench methodology - v3-style reports[/dim]",
            border_style="cyan",
        )
    )

    grid_given = any(k in overrides for k in ("preset", "batch_sizes", "io_lengths", "input_lengths", "output_lengths"))
    interactive = args.interactive or (not args.config and not grid_given and not args.yes and sys.stdin.isatty())
    if interactive:
        cfg = interactive_config(cfg, file_values, overrides)

    # Telemetry first so the idle baseline and the whole run are covered
    console.print("\n[yellow]Initializing system detection...[/yellow]")
    gpu = GPUTelemetry(cfg.gpu_poll_interval, cfg.gpu_indices, enabled=cfg.gpu_monitor)
    gpu.start()

    try:
        if cfg.engine == "endpoint":
            console.print(f"[yellow]Querying server at {cfg.url}...[/yellow]")
            engine = EndpointEngine(cfg)
        else:
            with console.status(f"[bold yellow]Loading {cfg.model} with vLLM LLM()...", spinner="dots"):
                engine = OfflineVLLMEngine(cfg)
        server_info = engine.discover()
    except (EngineError, ImportError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        gpu.stop()
        return 2

    poller = None
    fetcher = engine.metrics_fetcher() if cfg.server_metrics else None
    if fetcher:
        poller = ServerMetricsPoller(fetcher, poll=engine.kind == "endpoint")
        server_info.update(describe_server(poller.snapshot()))
    server_info["metrics_available"] = bool(poller and poller.available)
    for key in ("framework", "quantization", "tensor_parallel", "pipeline_parallel"):
        if getattr(cfg, key) is not None:
            server_info[key] = getattr(cfg, key)
    cfg.model = engine.model
    if not cfg.label:
        cfg.label = engine.model

    system_info = get_system_info(gpu.static_info())
    if cfg.hardware:
        system_info["gpu_description"] = cfg.hardware
    print_system_panels(system_info, server_info, gpu)

    factory = PromptFactory(cfg.seed, cfg.token_id_range)
    if cfg.prompt_format == "text":
        sample = PromptFactory.calibration_text(2000)
        tpw = engine.calibrate_tokens_per_word(sample, 2000)
        if tpw:
            factory.tokens_per_word = tpw
        console.print(f"[dim]Text prompts: {factory.tokens_per_word:.2f} tokens/word ({'calibrated via /tokenize' if tpw else 'default estimate'})[/dim]")

    # Preflight: one small request proves the prompt format and checks ignore_eos
    preflight: Dict[str, Any] = {}
    try:
        probe_prompt = (factory.token_prompts if cfg.prompt_format == "token_ids" else factory.text_prompts)(32, 16, 1, PHASE_PROBE, 999)[0]
        preflight = engine.preflight(probe_prompt)
    except EngineError as e:
        console.print(f"[bold red]Preflight failed:[/bold red] {e}")
        if cfg.prompt_format == "token_ids":
            console.print("[yellow]If the server rejects token-id prompts, rerun with --prompt-format text.[/yellow]")
        gpu.stop()
        return 2
    if preflight.get("ignore_eos_honoured") is False:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] asked for 16 tokens, got {preflight.get('completion_tokens')}. "
            "The server may ignore 'ignore_eos', so output lengths will vary."
        )
    else:
        console.print("[green]✓[/green] Preflight OK: exact-length generation (ignore_eos) confirmed")

    console.print()
    print_config_summary(cfg)
    if interactive and not args.yes and not Confirm.ask("\nProceed with this configuration?", default=True):
        console.print("[yellow]Benchmark cancelled.[/yellow]")
        gpu.stop()
        return 0

    idle_power = None
    if gpu.available and cfg.idle_power_seconds > 0:
        with console.status(f"[yellow]Measuring idle GPU power ({cfg.idle_power_seconds:.0f}s)...", spinner="dots"):
            idle_power = gpu.measure_idle_power(cfg.idle_power_seconds)
        if idle_power is not None:
            console.print(f"[dim]Idle power: {idle_power:.1f} W[/dim]")

    start_wall = datetime.now()
    stem = f"benchmark_v4_{sanitize_filename(cfg.label)}_{start_wall.strftime('%Y%m%d_%H%M%S')}"
    paths = output_paths(cfg.output_dir, stem)
    telemetry_meta = {
        "gpu_backend": gpu.backend_name,
        "gpu_poll_interval": cfg.gpu_poll_interval,
        "idle_power_w": idle_power,
        "server_metrics": server_info["metrics_available"],
    }

    passes = cfg.warmup_iterations + cfg.iterations + (1 if engine.kind == "offline" else 0)
    state = LiveState(cfg.test_queue(), passes, cfg.paired_io, gpu=gpu, poller=poller)
    state.label = cfg.label

    def checkpoint(r: SweepRunner) -> None:
        write_json(paths["json"], _build_package(cfg, r, system_info, server_info, telemetry_meta, preflight, start_wall, False))

    runner = SweepRunner(
        cfg,
        engine,
        factory,
        state,
        console,
        gpu=gpu,
        poller=poller,
        max_model_len=server_info.get("max_model_len"),
        idle_power=idle_power,
        on_checkpoint=checkpoint,
    )
    if poller:
        poller.start()

    console.print("\n[bold green]Starting benchmark execution...[/bold green]\n")
    interrupted = False
    try:
        if args.no_live or not console.is_terminal:
            runner.run()
        else:
            with Live(console=console, refresh_per_second=2, get_renderable=state.render):
                runner.run()
    except KeyboardInterrupt:
        interrupted = True
        runner.aborted = "Interrupted by user"
        console.print("\n[yellow]Interrupted - saving partial results...[/yellow]")
    finally:
        if poller:
            poller.stop()
        gpu.stop()
        engine.close()

    package = _build_package(cfg, runner, system_info, server_info, telemetry_meta, preflight, start_wall, not (interrupted or runner.aborted))
    write_json(paths["json"], package)
    write_results_csv(paths["csv"], runner.results)
    write_argonne_csv(paths["argonne_csv"], runner.results, package["metadata"])
    write_requests_csv(paths["requests_csv"], runner.request_rows)

    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print("[bold cyan]Benchmark Complete[/bold cyan]" if package["metadata"]["complete"] else "[bold yellow]Benchmark Incomplete[/bold yellow]")
    console.print(f"[bold green]{'=' * 80}[/bold green]\n")

    from .report import print_detailed_summary, print_highlights, render_report

    ok_rows = [r for r in runner.results if r.get("status") in ("ok", "partial")]
    if ok_rows:
        print_highlights(console, package)
    console.print("\n[bold cyan]Outputs:[/bold cyan]")
    console.print(f"  [green]*[/green] Results: [link=file://{paths['json'].resolve()}]{paths['json']}[/link]")
    console.print(f"  [green]*[/green] CSV: {paths['csv']}")
    console.print(f"  [green]*[/green] LLM-Inference-Bench CSV: {paths['argonne_csv']}")
    if runner.request_rows:
        console.print(f"  [green]*[/green] Per-request CSV: {paths['requests_csv']}")
    if ok_rows:
        console.print("\n[yellow]Generating visualizations...[/yellow]")
        png = render_report(package, paths["png"], dpi=cfg.png_dpi)
        console.print(f"  [green]*[/green] Charts: [link=file://{png.resolve()}]{png}[/link]")
        if args.details or (interactive and Confirm.ask("\n[cyan]Show detailed summary table?[/cyan]", default=False)):
            print_detailed_summary(console, package)

    total = package["metadata"]["benchmark_duration"]
    measured = sum(r.get("total_time", 0) for r in runner.results)
    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print(f"[bold cyan]Total Benchmark Time: {total / 60:.1f} minutes ({total:.0f} seconds)[/bold cyan]")
    console.print(f"[dim]  Measured batches: {fmt_duration(measured)} | Warmup, probes, pauses: {fmt_duration(max(0.0, total - measured))}[/dim]")
    console.print(f"[bold green]{'=' * 80}[/bold green]")
    return 0 if package["metadata"]["complete"] else 1


def cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .report import print_detailed_summary, print_highlights, render_report

    status = 0
    for path in args.json:
        with open(path) as f:
            package = json.load(f)
        if package.get("metadata", {}).get("suite_version", "").split(".")[0] != "4":
            console.print(f"[red]{path} is not a v4 results file[/red]")
            status = 1
            continue
        out = Path(args.output) if args.output and len(args.json) == 1 else Path(path).with_suffix(".png")
        png = render_report(package, out, dpi=args.dpi)
        console.print(f"[green]*[/green] Charts: {png}")
        print_highlights(console, package)
        if args.details:
            print_detailed_summary(console, package)
    return status


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "report", "-h", "--help", "--version"):
        argv.insert(0, "run")

    parser = argparse.ArgumentParser(
        prog="vllm_benchmark_suitev4.py",
        description="vLLM Benchmark Suite v4: LLM-Inference-Bench methodology with v3-style reports.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Run a benchmark (default)")
    _add_run_args(run_p)
    rep_p = sub.add_parser("report", help="Re-render the PNG report from v4 results JSON")
    rep_p.add_argument("json", nargs="+")
    rep_p.add_argument("--output", help="PNG path (single input only)")
    rep_p.add_argument("--dpi", type=int, default=300)
    rep_p.add_argument("--details", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            return cmd_report(args)
        return cmd_run(args)
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
