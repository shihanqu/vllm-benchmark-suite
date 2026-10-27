import numpy as np
import pytest

from suite_v4.config import PRESETS, build_config, load_config_file, parse_int_list
from suite_v4.workload import PHASE_MEASURE, PHASE_WARMUP, PromptFactory


def test_argonne_preset_matches_llm_inference_bench():
    cfg = build_config(overrides={"preset": "argonne"})
    assert cfg.io_lengths == [128, 256, 512, 1024, 2048]
    assert cfg.batch_sizes == [1, 16, 32, 64]
    assert (cfg.warmup_iterations, cfg.iterations) == (10, 30)
    assert cfg.temperature == 1.0 and cfg.top_p == 1.0 and cfg.ignore_eos
    assert cfg.shapes() == [(n, n) for n in cfg.io_lengths]
    assert len(cfg.test_queue()) == 20


def test_precedence_and_grid_exclusivity(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[workload]\npreset = "quick"\nbatch_sizes = [2, 4]\n[endpoint]\nurl = "http://h:1/"\n')
    file_values = load_config_file(str(f))
    cfg = build_config(file_values=file_values, overrides={"iterations": 7})
    assert cfg.preset == "quick" and cfg.batch_sizes == [2, 4] and cfg.iterations == 7
    assert cfg.url == "http://h:1"
    # Setting only output lengths keeps the preset's lengths as inputs (cross grid)
    cfg = build_config(overrides={"output_lengths": [500]})
    assert cfg.io_lengths is None
    assert cfg.input_lengths == PRESETS["standard"]["io_lengths"] and cfg.output_lengths == [500]
    # io_lengths in a later layer wins over a cross grid from the preset
    cfg = build_config(overrides={"preset": "long-context", "io_lengths": [64]})
    assert cfg.shapes() == [(64, 64)]


def test_unknown_key_rejected(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"batch_size": [1]}')
    with pytest.raises(ValueError, match="batch_size"):
        load_config_file(str(f))


def test_chat_forces_text_and_seed_is_generated():
    cfg = build_config(overrides={"api": "chat"})
    assert cfg.prompt_format == "text"
    a, b = build_config(), build_config()
    assert isinstance(a.seed, int) and a.seed != b.seed  # fresh prompts per run (no prefix-cache reuse)
    assert build_config(overrides={"seed": 5}).seed == 5


def test_parse_int_list():
    assert parse_int_list("1,16, 32") == [1, 16, 32]
    assert parse_int_list("1K,4k,2Ki") == [1000, 4000, 2048]
    assert parse_int_list(None) is None


def test_token_prompts_exact_unique_reproducible():
    f = PromptFactory(seed=3, token_id_range=(1000, 10000))
    p = f.token_prompts(128, 128, 4, PHASE_MEASURE, 0)
    assert len(p) == 4 and all(len(x) == 128 for x in p)
    assert all(1000 <= int(v) < 10000 for x in p for v in x)
    assert not np.array_equal(p[0], p[1])  # unique per request
    assert not np.array_equal(p[0], f.token_prompts(128, 128, 4, PHASE_MEASURE, 1)[0])  # unique per iteration
    assert not np.array_equal(p[0], f.token_prompts(128, 128, 4, PHASE_WARMUP, 0)[0])  # warmup != measured
    assert np.array_equal(p[0], PromptFactory(seed=3).token_prompts(128, 128, 4, PHASE_MEASURE, 0)[0])


def test_text_prompts_sized_by_tokens_per_word():
    f = PromptFactory(seed=1, tokens_per_word=2.0)
    p = f.text_prompts(200, 10, 2, PHASE_MEASURE, 0)
    assert len(p) == 2 and len(p[0].split()) == 100
