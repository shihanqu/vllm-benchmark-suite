#!/usr/bin/env python3
"""
Non-interactive benchmark runner.
Patches vllm_benchmark_suitev2 to use port 1235 and custom parameters,
waits 120s for model to finish loading, then runs.
"""
import time
import sys

# --- Wait skipped: model should be loaded by now (120s+ has elapsed) ---
print("[RUNNER] Starting benchmark (wait already elapsed).\n")

# --- Patch the module before importing the rest ---
import vllm_benchmark_suitev2 as bench

# Override API URL to port 1235
bench.API_BASE_URL = "http://localhost:1235"
bench.API_ENDPOINT = f"{bench.API_BASE_URL}/v1/chat/completions"
bench.API_MODELS_ENDPOINT = f"{bench.API_BASE_URL}/v1/models"
bench.API_HEALTH_ENDPOINT = f"{bench.API_BASE_URL}/health"
bench.API_VERSION_ENDPOINT = f"{bench.API_BASE_URL}/version"

# Override the interactive config function to return our custom values
_original_get_interactive_config = bench.get_interactive_config

def custom_config():
    context_lengths = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    concurrent_users = [1, 2, 4, 8, 16, 32, 64]
    output_tokens = 500
    prompt_types = ["classic"]

    # Print the config for visibility
    from rich.table import Table
    from rich import box
    bench.console.print("\n[bold cyan]••• Benchmark Configuration (Non-Interactive) •••[/bold cyan]\n")
    config_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    config_table.add_row("Context Lengths", ", ".join([f"{c//1000}K" for c in context_lengths]))
    config_table.add_row("Concurrent Users", ", ".join([str(u) for u in concurrent_users]))
    config_table.add_row("Output Tokens", str(output_tokens))
    config_table.add_row("Prompt Types", ", ".join(prompt_types))
    config_table.add_row("Total Tests", str(len(context_lengths) * len(concurrent_users) * len(prompt_types)))
    est_time = len(context_lengths) * len(concurrent_users) * len(prompt_types) * 30
    config_table.add_row("Est. Duration", f"{est_time // 60} min")
    bench.console.print(config_table)
    bench.console.print()

    return context_lengths, concurrent_users, output_tokens, prompt_types

bench.get_interactive_config = custom_config

# Also patch the final "Show detailed summary table?" confirm to auto-skip
import rich.prompt
_original_confirm_ask = rich.prompt.Confirm.ask
def auto_confirm(*args, **kwargs):
    return False
rich.prompt.Confirm.ask = auto_confirm

# --- Run ---
if __name__ == "__main__":
    bench.main()
