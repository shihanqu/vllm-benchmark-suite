# vLLM Benchmark Suite

Benchmark OpenAI-compatible LLM inference servers (vLLM, SGLang, ...) on your own GPUs. It measures throughput, latency, time to first token (TTFT), inter-token latency, energy per token and server internals across batch sizes and sequence lengths, and renders everything as a one-page dashboard PNG.

**v4 is the main benchmark.** It measures with the method of [LLM-Inference-Bench](https://github.com/argonne-lcf/LLM-Inference-Bench) from Argonne National Laboratory ([paper](https://arxiv.org/abs/2411.00136)), so every run puts the same standardized load on the server and produces numbers you can compare across models, servers, GPUs and power limits. See [Credits](#credits).

Published results: [wiki](https://github.com/shihanqu/vllm-benchmark-suite/wiki)

## Why v4 is the main benchmark

v1-v3 were good for a quick look at one server, but their numbers don't compare well between runs:

- **Prompts were estimated and shared.** Prompt length was guessed at ~4 characters per token, and every concurrent user sent the *same* prompt. With prefix caching on, the server skipped most of the prompt processing (v3's own cache heatmaps show 80-99% hits), so TTFT and long-context throughput looked better than real traffic would get.
- **Output length varied.** Generation stopped at EOS, so two models could do different amounts of work in the same test.
- **Several metrics were estimates.** TTFT, inter-token latency and the prefill/decode split were fixed fractions (10-15%) of the total request time.

v4 replaces all of that with the LLM-Inference-Bench method:

- **Identical, exact workload.** Every request gets random token-id prompts of exactly the requested length, fresh for every batch, and generates exactly the requested number of tokens (`ignore_eos`). A preflight request checks that the server honours this.
- **Measured, not estimated.** TTFT, per-token latency and stream stalls come from the token stream of every request.
- **Comparable definitions.** Latency and throughput use the paper's definitions, and one CSV uses the paper's column layout, so results line up with its published data for the same model and settings.
- **Power studies built in.** The GPU energy counter, enforced power limit and time spent power-capped or thermally throttled are recorded for every test.
- **Reproducible and robust.** Runs are driven by config files and record their seed. Results are checkpointed after every test, so a crash or OOM still leaves a usable report.

## Quick start

```bash
pip install -r requirements.txt

python vllm_benchmark_suitev4.py                                   # interactive, like v3
python vllm_benchmark_suitev4.py run --config configs/v4/example.toml
python vllm_benchmark_suitev4.py run --url http://localhost:8000 --preset quick -y
./run_benchmark_v4.sh configs/v4/example.toml                      # creates a venv if needed, then runs

python vllm_benchmark_suitev4.py report outputs/benchmark_v4_<label>_<timestamp>.json   # re-render a PNG
```

Start your server first (for example `vllm serve <model> --port 8000`). v4 talks to `/v1/completions`; use `--api chat` for servers that only offer chat.

## How v4 measures

| LLM-Inference-Bench (vLLM scripts) | v4 |
|---|---|
| Batch of B prompts in one `llm.generate()` | **Endpoint (default):** B streamed requests released together. **Offline:** `vllm.LLM().generate()` in-process, as in the paper |
| Dummy prompts: random token ids, exactly `input_len` long | Same (ids in [1000, 10000) to avoid control tokens), drawn fresh for every batch so the prefix cache can't skip prefill |
| `ignore_eos=True`, `max_tokens=output_len` | Same, checked by a preflight request |
| `temperature=1.0`, `top_p=1.0` | Same (configurable) |
| 10 warmup + 30 measured batches | `--preset argonne`; the default `standard` preset uses 2 + 5 |
| Grid: batch 1/16/32/64 × input = output 128..2048 | Same grid by default; any `io_lengths`, or `input_lengths` × `output_lengths` |
| Latency = mean batch wall time (p10..p99 reported) | Same |
| Throughput = B × (input + output) / Latency | Same, plus generation throughput (v3's tokens/sec) |
| Power: NVML every 0.25 s, throughput per watt | NVML every 0.1 s plus the GPU **energy counter** (joules per token); nvidia-smi fallback |
| TTFT = latency of a `max_tokens=1` batch | Offline: same. Endpoint: TTFT, TPOT and ITL measured per request |
| CSV: `Hardware, Num of Hardware, Framework, Model, Input Output Length, Batch Size, Latency, Throughput, ...` | `*_llm-inference-bench.csv` uses those columns |

Each run also reads the server's `/metrics` endpoint (vLLM, with SGLang fallbacks): KV cache usage, queueing, preemptions, prefix-cache hits and speculative-decoding acceptance. It records idle power before the run.

## What you get

| File in `./outputs` | Contents |
|---|---|
| `benchmark_v4_<label>_<ts>.png` | Dashboard (below) |
| `benchmark_v4_<label>_<ts>.json` | Metadata (system, server, config, methodology, telemetry timeline) and one row per test. Rows also carry v3 column names (`context_length`, `concurrent_users`, `tokens_per_second`), so `consolidate_data.py`-style scripts keep working |
| `benchmark_v4_<label>_<ts>.csv` | All per-test metrics |
| `benchmark_v4_<label>_<ts>_llm-inference-bench.csv` | LLM-Inference-Bench column layout |
| `benchmark_v4_<label>_<ts>_requests.csv` | Per request: TTFT, TPOT, max ITL, end-to-end time, tokens, errors |

The dashboard keeps the v3 layout and style. The x-axis is input length (log scale) and each line is a batch size (v3's "concurrent users"):

1. **Headline numbers:** peak generation and total throughput, single-stream decode speed, best TTFT, best tokens per joule, run summary
2. Generation throughput vs length, with a p10-p90 band across iterations
3. **Total throughput** and **batch latency** heatmaps (LLM-Inference-Bench definitions)
4. Throughput per user · TTFT (measured p50) · request latency
5. Prompt processing speed · inter-token latency (measured) · decode speed
6. Power (with power-limit and idle lines) · SM clock (with max clock) · batch scaling efficiency
7. **Energy:** joules per output token · tokens per joule · % of time power-capped
8. **Tail latency:** TTFT p99 · ITL p99 (stream stalls) · run-to-run variability
9. **Server internals:** peak KV cache · queue time · preemptions · speculative-decoding acceptance · prefix-cache hit rate (only if > 1%) · tokens per streamed chunk
10. **Run timeline:** GPU power and SM clock over the whole run, with each test's measured window shaded
11. **Input × output heatmaps** when testing several output lengths, like the paper's input-vs-output figure

The header shows the GPU, power limit, driver, server version, quantization, KV cache capacity, prefix caching, speculative decoding and the method used. Failed tests (e.g. OOM) and skipped ones (longer than `max_model_len`) are marked `FAIL` / `SKIP`.

## Presets and configuration

| Preset | Grid | Warmup + measured batches |
|---|---|---|
| `standard` (default) | batch 1, 16, 32, 64 × input = output 128..2048 | 2 + 5 |
| `argonne` | same grid | 10 + 30 (the paper's settings) |
| `long-context` | batch 1..64 × input 1K..256K, output 500 (the v3 grid) | 1 + 3 |
| `quick` | batch 1, 16, 64 × input = output 128 / 512 / 2048 | 1 + 2 |

Settings resolve as preset < config file (`configs/v4/*.toml` or `.json`) < CLI flags. `configs/v4/example.toml` documents every key, and `python vllm_benchmark_suitev4.py run --help` lists every flag. Commonly used:

- `--label` sets the title and file name.
- `--gpu-indices 0,1` picks the GPUs that serve the model.
- `--output-lengths 500` keeps the preset's input lengths and fixes the output.
- `--api chat` sends text prompts to chat-only servers.
- `--engine offline --model <hf-id> --tp N` runs vLLM in-process like the paper.

## Comparing with v1-v3 results

- Expect **higher TTFT and lower long-context throughput** than v3 for the same server. v4 numbers are real; v3's were helped by prefix caching.
- v4 always generates exactly `output_len` tokens; v3 let generations stop early.
- **Speculative decoding** (MTP, DFlash, EAGLE) accepts fewer draft tokens on random-token prompts than on natural text, so v4 shows close to worst-case decode speed for those setups. The acceptance-length panel shows what was achieved.
- Text prompts (`--prompt-format text` or `--api chat`) are sized with vLLM's `/tokenize` when available and are approximate. Token-id prompts are exact.

## Testing without a GPU

`tests/v4/mock_server.py` simulates a continuous-batching OpenAI-compatible server: KV capacity, queueing, preemption, optional speculative decoding and vLLM-style `/metrics`. `tests/v4/fake_nvidia_smi.py` reports power-limited GPUs that follow its load.

```bash
python tests/v4/mock_server.py --port 8123 --speed 10 &
python vllm_benchmark_suitev4.py run --url http://localhost:8123 --preset quick -y
pip install pytest && python -m pytest tests/v4
```

## Versions

| Version | Entry point | What it is | Docs |
|---|---|---|---|
| **v4 (current)** | `vllm_benchmark_suitev4.py`, `run_benchmark_v4.sh` | LLM-Inference-Bench method, measured streaming metrics, energy and server internals | This file, [CHANGELOG](CHANGELOG.md) |
| v3 | `vllm_benchmark_suitev3.py`, `run_benchmark_v3.py`, `run_benchmark.sh` | v2 plus prompt variety, `/metrics` cache stats, CSV export | [README_v2.md](README_v2.md) |
| v2 | `vllm_benchmark_suitev2.py` | Rich UI, server auto-detection, energy metrics | [README_v2.md](README_v2.md) |
| v1 | `vllm_benchmark_suite.py` | Original release | [README_v1.md](README_v1.md) |

Older versions are kept unchanged. Files in `outputs/` named `benchmark_v4_*` come from v4; the others come from v1-v3.

## Credits

### LLM-Inference-Bench

v4's measurement method comes from **LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators** by Krishna Teja Chitty-Venkata, Siddhisanket Raskar, Bharat Kale, Farah Ferdaus, Aditya Tanikanti, Ken Raffenetti, Valerie Taylor, Murali Emani and Venkatram Vishwanath (Argonne National Laboratory).

- Code: [github.com/argonne-lcf/LLM-Inference-Bench](https://github.com/argonne-lcf/LLM-Inference-Bench) (BSD-3-Clause)
- Paper: [arXiv:2411.00136](https://arxiv.org/abs/2411.00136)
- Results dashboard: [argonne-lcf.github.io/LLM-Inference-Bench](https://argonne-lcf.github.io/LLM-Inference-Bench/)

v4 adopts its workload (random token-id prompts, forced output length, batch × input/output-length grid), sampling settings, warmup and measured loop, latency and throughput definitions, throughput per watt, `max_tokens=1` TTFT probe and CSV layout. v4 reimplements these in its own code; it does not include LLM-Inference-Bench code. If you publish results produced with v4, please cite the paper:

```bibtex
@article{chitty2024llm,
  title={LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators},
  author={Chitty-Venkata, Krishna Teja and Raskar, Siddhisanket and Kale, Bharat and Ferdaus, Farah and Tanikanti, Aditya and Raffenetti, Ken and Taylor, Valerie and Emani, Murali and Vishwanath, Venkatram},
  journal={arXiv preprint arXiv:2411.00136},
  year={2024}
}
```

### This suite

- Original suite (v1, v2): Amit Lokhande ([notaDestroyer](https://github.com/notaDestroyer))
- Prompt variety, vLLM `/metrics` tracking and CSV export: Sean Steinle ([SeanSteinle](https://github.com/SeanSteinle))
- v3, v4 and the published results: Shihan Qu ([shihanqu](https://github.com/shihanqu))
- Built on [vLLM](https://github.com/vllm-project/vllm), [Rich](https://github.com/Textualize/rich), [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)

## License

MIT, see [LICENSE](LICENSE).
