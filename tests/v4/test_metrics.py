import pytest

from suite_v4.engines import BatchResult, RequestRecord, infer_quantization
from suite_v4.metrics import add_scaling_efficiency, aggregate_test
from suite_v4.servermetrics import derive_window, describe_server, parse_prometheus


def _batch(start, latency, n, in_len, out_len, ttft, chunk_gap):
    recs = []
    for i in range(n):
        e2e = ttft + chunk_gap * (out_len - 1)
        recs.append(
            RequestRecord(
                ok=True,
                start=start,
                ttft=ttft + 0.001 * i,
                e2e=e2e + 0.001 * i,
                itls=[chunk_gap] * (out_len - 1),
                chunks=out_len,
                prompt_tokens=in_len,
                completion_tokens=out_len,
            )
        )
    return BatchResult(start=start, end=start + latency, records=recs)


def test_llm_inference_bench_formulas():
    its = [_batch(0.0, 2.0, 4, 128, 128, 0.05, 0.015), _batch(10.0, 2.2, 4, 128, 128, 0.05, 0.015)]
    row = aggregate_test(128, 128, 4, its, gpu_stats={"avg_power": 400.0}, energy_counter_j=840.0, window_seconds=4.2)
    assert row["status"] == "ok"
    assert row["latency_avg"] == pytest.approx(2.1)
    # Throughput = B * (in + out) / Latency
    assert row["throughput"] == pytest.approx(4 * 256 / 2.1)
    assert row["output_throughput"] == pytest.approx(4 * 128 / 2.1)
    assert row["throughput_per_watt"] == pytest.approx(row["throughput"] / 400.0)
    # TPOT = (e2e - ttft) / (tokens - 1)
    assert row["tpot_mean_ms"] == pytest.approx(15.0, rel=1e-6)
    assert row["itl_p50_ms"] == pytest.approx(15.0)
    assert row["output_token_fidelity"] == pytest.approx(100.0)
    # Energy from the counter: 840 J for 2 * 4 * 128 generated tokens
    assert row["energy_per_output_token"] == pytest.approx(840.0 / 1024)
    assert row["tokens_per_joule"] == pytest.approx(1024 / 840.0)
    # Prefill throughput = batch prompt tokens / time to the last first token
    assert row["prefill_throughput"] == pytest.approx(4 * 128 / 0.053)
    # v3 aliases
    assert row["tokens_per_second"] == row["output_throughput"] and row["concurrent_users"] == 4


def test_failed_and_partial_rows():
    bad = BatchResult(0, 1, [RequestRecord(ok=False, error="HTTP 500")])
    assert aggregate_test(64, 64, 1, [bad])["status"] == "failed"
    mixed = _batch(0.0, 1.0, 2, 64, 64, 0.01, 0.01)
    mixed.records.append(RequestRecord(ok=False, error="HTTP 500"))
    row = aggregate_test(64, 64, 3, [mixed])
    assert row["status"] == "partial" and row["success_rate"] == pytest.approx(200 / 3)


def test_offline_probe_ttft():
    recs = [RequestRecord(ok=True, start=0, e2e=3.0, prompt_tokens=512, completion_tokens=512) for _ in range(8)]
    row = aggregate_test(512, 512, 8, [BatchResult(0, 3.0, recs)], probe_latency=0.2)
    assert row["ttft_source"] == "probe" and row["ttft_mean_ms"] == pytest.approx(200.0)
    assert row["tpot_mean_ms"] == pytest.approx(2800.0 / 511)


def test_scaling_efficiency():
    rows = [
        {"input_len": 1, "output_len": 1, "batch_size": 1, "output_throughput": 100.0},
        {"input_len": 1, "output_len": 1, "batch_size": 4, "output_throughput": 200.0},
    ]
    add_scaling_efficiency(rows)
    assert rows[0]["batch_efficiency"] == pytest.approx(100.0)
    assert rows[1]["batch_efficiency"] == pytest.approx(50.0)


METRICS_BEFORE = """
# HELP vllm:cache_config_info x
vllm:cache_config_info{block_size="16",enable_prefix_caching="False",gpu_memory_utilization="0.92",num_gpu_blocks="1000"} 1.0
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_hits_total{engine="0"} 10
vllm:num_preemptions_total{engine="0"} 1
vllm:spec_decode_num_drafts_total 10
vllm:spec_decode_num_draft_tokens_total 30
vllm:spec_decode_num_accepted_tokens_total 15
vllm:request_queue_time_seconds_sum{model_name="m"} 1.0
vllm:request_queue_time_seconds_count{model_name="m"} 2
vllm:request_queue_time_seconds_bucket{le="+Inf",model_name="m"} 2
vllm:gpu_cache_usage_perc 0.25
"""

METRICS_AFTER = """
vllm:prefix_cache_queries_total{engine="0"} 300
vllm:prefix_cache_hits_total{engine="0"} 10
vllm:num_preemptions_total{engine="0"} 4
vllm:spec_decode_num_drafts_total 110
vllm:spec_decode_num_draft_tokens_total 330
vllm:spec_decode_num_accepted_tokens_total 165
vllm:request_queue_time_seconds_sum{model_name="m"} 3.0
vllm:request_queue_time_seconds_count{model_name="m"} 6
vllm:gpu_cache_usage_perc 0.5
"""


def test_prometheus_window_and_server_description():
    before, after = parse_prometheus(METRICS_BEFORE), parse_prometheus(METRICS_AFTER)
    gauges = [{"kv_usage": after.gauge("kv_usage"), "running": 3.0, "waiting": 1.0}]
    out = derive_window(before, after, gauges)
    assert out["prefix_cache_hit_rate"] == pytest.approx(0.0)
    assert out["preemptions"] == 3
    assert out["spec_acceptance_rate"] == pytest.approx(50.0)
    assert out["spec_mean_acceptance_length"] == pytest.approx(2.5)
    assert out["server_queue_time_mean"] == pytest.approx(0.5)
    assert out["kv_cache_usage_peak"] == pytest.approx(50.0)  # fraction converted to percent
    info = describe_server(before)
    assert info["kv_cache_capacity_tokens"] == 16000
    assert info["prefix_caching"] is False and info["gpu_memory_utilization"] == pytest.approx(0.92)
    assert info["speculative_decoding"] is True


def test_infer_quantization():
    assert infer_quantization("Qwen3.6-35B-A3B-NVFP4") == "NVFP4"
    assert infer_quantization("Qwen/Qwen3-30B-A3B-Instruct-FP8") == "FP8"
    assert infer_quantization("org/model-AWQ") == "AWQ"
    assert infer_quantization("meta-llama/Llama-3-8B") is None
