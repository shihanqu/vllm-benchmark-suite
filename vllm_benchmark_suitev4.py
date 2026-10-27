#!/usr/bin/env python3
"""
vLLM Performance Benchmark Suite v4

LLM-Inference-Bench methodology (Argonne National Laboratory) under the hood:
fixed-size batches of random token-id prompts with forced output length, warmup + measured
iterations, batch latency / throughput / power exactly as the paper defines them - plus
streamed TTFT/ITL, energy counters and server internals, rendered in the v3 report style.

Usage:
    python vllm_benchmark_suitev4.py                      # interactive (like v3)
    python vllm_benchmark_suitev4.py run --config configs/v4/example.toml
    python vllm_benchmark_suitev4.py report outputs/benchmark_v4_<label>_<timestamp>.json

See README.md (v4 section) for all options.
"""

import sys

from suite_v4.cli import main

if __name__ == "__main__":
    sys.exit(main())
