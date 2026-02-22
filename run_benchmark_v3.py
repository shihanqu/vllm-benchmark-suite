#!/usr/bin/env python3
"""
Non-interactive benchmark runner for v3.
Patches vllm_benchmark_suitev3 to use port 1235 and custom parameters,
then runs the benchmark non-interactively.
"""
import sys

print("[RUNNER] Starting v3 benchmark (non-interactive).\n")

# --- Patch the module before importing the rest ---
import vllm_benchmark_suitev3 as bench

# Ensure API URL points to port 1235
bench.API_BASE_URL = "http://localhost:1235"
bench.API_ENDPOINT = f"{bench.API_BASE_URL}/v1/chat/completions"
bench.API_MODELS_ENDPOINT = f"{bench.API_BASE_URL}/v1/models"
bench.API_HEALTH_ENDPOINT = f"{bench.API_BASE_URL}/health"
bench.API_VERSION_ENDPOINT = f"{bench.API_BASE_URL}/version"

# ============================================================
# GPU Description Override
# Set this to override the auto-detected GPU name in reports
# ============================================================
GPU_DESCRIPTION = "1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96GB) - 450W Limit"

# Patch SystemInfo.get_system_info to inject the GPU override
_original_get_system_info = bench.SystemInfo.get_system_info

@staticmethod
def patched_get_system_info():
    info = _original_get_system_info()
    if GPU_DESCRIPTION:
        info["gpu_name"] = GPU_DESCRIPTION
    return info

bench.SystemInfo.get_system_info = patched_get_system_info

# Override the interactive config function to return our custom values
_original_get_interactive_config = bench.get_interactive_config

def custom_config():
    context_lengths = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
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
    ctx_labels = [f"{c//1000}K" if c >= 1000 else str(c) for c in context_lengths]
    config_table.add_row("Context Lengths", ", ".join(ctx_labels))
    config_table.add_row("Concurrent Users", ", ".join([str(u) for u in concurrent_users]))
    config_table.add_row("Output Tokens", str(output_tokens))
    config_table.add_row("Prompt Types", ", ".join(prompt_types))
    config_table.add_row("GPU Override", GPU_DESCRIPTION or "(auto-detect)")
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
