"""Full runs against the mock server / fake vllm, checking every output file."""

import csv
import io
import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from suite_v4 import cli
from suite_v4.dashboard import LiveState
from suite_v4.output import ARGONNE_COLUMNS

HERE = Path(__file__).resolve().parent


def _run(tmp_path, *args):
    out = tmp_path / "out"
    code = cli.main(["run", *args, "-y", "--no-live", "--cooldown", "0", "--output-dir", str(out), "--dpi", "40"])
    files = {}
    for p in out.iterdir() if out.exists() else []:
        key = p.stem.rsplit("_", 1)[1] if p.stem.endswith(("_requests", "_llm-inference-bench")) else p.suffix
        files[key] = p
    return code, files


def test_endpoint_run_outputs(mock_server, tmp_path):
    code, files = _run(tmp_path, "--url", mock_server, "--io-lengths", "64,256", "--batch-sizes", "1,8",
                       "--warmup", "1", "--iterations", "2", "--label", "e2e")
    assert code == 0
    pkg = json.loads(files[".json"].read_text())
    meta, rows = pkg["metadata"], pkg["results"]
    assert meta["complete"] and meta["suite_version"].startswith("4.")
    assert meta["preflight"]["ignore_eos_honoured"] is True
    assert meta["telemetry"]["gpu_backend"] == "nvidia-smi"
    assert meta["system_info"]["power_limits_w"] == [300.0, 300.0]
    assert meta["server_info"]["kv_cache_capacity_tokens"] == 120000
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    for r in rows:
        assert r["output_token_fidelity"] == pytest.approx(100.0)  # ignore_eos honoured
        assert r["prompt_tokens_mean"] == r["input_len"]  # exact token-id prompts
        assert r["prefix_cache_hit_rate"] == pytest.approx(0.0)  # fresh prompts every iteration
        assert r["ttft_source"] == "stream" and r["ttft_p50_ms"] > 0 and r["tpot_mean_ms"] > 0
        assert r["avg_power"] > 0 and r["energy_per_output_token"] > 0
        assert r["throughput"] == pytest.approx(r["batch_size"] * 2 * r["input_len"] / r["latency_avg"], rel=0.01)
    with open(files["llm-inference-bench"]) as f:
        reader = list(csv.reader(f))
    assert reader[0] == ARGONNE_COLUMNS and len(reader) == 5
    assert reader[1][0] == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" and reader[1][1] == "2"
    assert files[".png"].stat().st_size > 10000
    assert sum(1 for _ in open(files["requests"])) == 1 + 2 * (1 + 8) * 2  # header + per-request rows

    # Re-render from JSON
    png = tmp_path / "again.png"
    assert cli.main(["report", str(files[".json"]), "--output", str(png), "--dpi", "40"]) == 0
    assert png.stat().st_size > 10000


def test_skip_fail_and_spec_decode(mock_server_factory, tmp_path):
    _, url = mock_server_factory("--spec", "--max-model-len", "2048", "--kv-capacity", "20000")
    code, files = _run(tmp_path, "--url", url, "--input-lengths", "256,1K,4K", "--output-lengths", "64,128",
                       "--batch-sizes", "1,24", "--warmup", "0", "--iterations", "1")
    rows = json.loads(files[".json"].read_text())["results"]
    status = {(r["input_len"], r["output_len"], r["batch_size"]): r["status"] for r in rows}
    assert status[(4000, 64, 1)] == "skipped"
    assert status[(256, 64, 24)] == "ok"
    ok = [r for r in rows if r["status"] == "ok"]
    assert all(r["tokens_per_chunk"] > 1.5 for r in ok)  # multi-token chunks from speculative decoding
    assert all(r["spec_mean_acceptance_length"] > 1.5 for r in ok)
    assert any(r.get("waiting_peak", 0) > 0 or r.get("preemptions", 0) > 0 for r in ok)  # KV pressure at batch 24
    assert files[".png"].exists() and code == 0

    # Without the skip, over-long shapes are sent and fail cleanly
    code, files = _run(tmp_path / "b", "--url", url, "--io-lengths", "4K", "--batch-sizes", "1",
                       "--warmup", "0", "--iterations", "1", "--no-skip-context")
    rows = json.loads(files[".json"].read_text())["results"]
    assert rows[0]["status"] == "failed" and "HTTP 400" in rows[0]["error_sample"]
    assert ".png" not in files  # nothing succeeded, so no chart


def test_chat_text_prompts(mock_server, tmp_path):
    code, files = _run(tmp_path, "--url", mock_server, "--api", "chat", "--io-lengths", "128", "--batch-sizes", "2",
                       "--warmup", "0", "--iterations", "1", "--no-gpu-monitor")
    row = json.loads(files[".json"].read_text())["results"][0]
    assert code == 0 and row["status"] == "ok"
    assert abs(row["prompt_tokens_mean"] - 128) < 30  # calibrated via /tokenize (+ chat template)
    assert row["completion_tokens_mean"] == 128


def test_server_down_aborts_with_partial_results(fake_gpu, tmp_path):
    code, files = _run(tmp_path, "--url", "http://127.0.0.1:9", "--io-lengths", "64", "--batch-sizes", "1")
    assert code == 2  # discovery fails before any test runs
    assert not files


def test_offline_engine_with_fake_vllm(fake_gpu, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE / "fake_vllm"))
    sys.modules.pop("vllm", None)
    code, files = _run(tmp_path, "--engine", "offline", "--model", "fake/Llama-3-8B", "--tp", "2", "--dtype", "float16",
                       "--io-lengths", "64,128", "--batch-sizes", "1,4", "--warmup", "1", "--iterations", "2",
                       "--engine-arg", "enforce_eager=true")
    import vllm

    assert code == 0
    assert vllm.LLM.last_kwargs["tensor_parallel_size"] == 2 and vllm.LLM.last_kwargs["enforce_eager"] is True
    pkg = json.loads(files[".json"].read_text())
    assert pkg["metadata"]["server_info"]["framework"].startswith("vLLM (offline")
    for r in pkg["results"]:
        assert r["status"] == "ok" and r["ttft_source"] == "probe"
        assert r["output_token_fidelity"] == pytest.approx(100.0)
        assert r["prefix_cache_hit_rate"] == pytest.approx(0.0)  # via llm.get_metrics()
    sys.modules.pop("vllm", None)


def test_dashboard_renders_in_every_phase():
    state = LiveState([(128, 128, 1), (128, 128, 16), (2048, 500, 64)], passes=3, paired=False)
    console = Console(file=io.StringIO(), width=120, height=50, force_terminal=True, color_system=None)
    console.print(state.render())
    state.begin_test(0, (128, 128, 1))
    for phase in ("warmup", "probe", "measure"):
        state.set_phase(phase, 1, 2)
        console.print(state.render())
    state.end_test({"status": "ok", "input_len": 128, "output_len": 128, "batch_size": 1, "output_throughput": 10.0,
                    "throughput": 20.0, "latency_avg": 1.0, "ttft_p50_ms": 5.0, "tpot_mean_ms": 2.0})
    state.end_test({"status": "failed", "input_len": 128, "output_len": 128, "batch_size": 16})
    console.print(state.render())
    text = console.file.getvalue()
    assert "OVERALL PROGRESS" in text and "FAILED" in text and "2K→500" in text
