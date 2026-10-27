"""
Benchmark configuration: presets, config files (TOML/JSON) and CLI overrides.

Precedence (lowest to highest): dataclass defaults < preset < config file < CLI flags.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Workload grids. "argonne" reproduces the LLM-Inference-Bench vLLM sweep exactly
# (vLLM/H100/run-throughput-bench.sh + benchmark_latency.py defaults: 10 warmup, 30 measured).
PRESETS: Dict[str, Dict[str, Any]] = {
    "argonne": {
        "io_lengths": [128, 256, 512, 1024, 2048],
        "batch_sizes": [1, 16, 32, 64],
        "warmup_iterations": 10,
        "iterations": 30,
    },
    "standard": {
        "io_lengths": [128, 256, 512, 1024, 2048],
        "batch_sizes": [1, 16, 32, 64],
        "warmup_iterations": 2,
        "iterations": 5,
    },
    "long-context": {
        # Same grid as run_benchmark_v3.py, measured with the v4 methodology
        "input_lengths": [1000, 4000, 16000, 48000, 128000, 256000],
        "output_lengths": [500],
        "batch_sizes": [1, 2, 4, 8, 16, 32, 64],
        "warmup_iterations": 1,
        "iterations": 3,
    },
    "quick": {
        "io_lengths": [128, 512, 2048],
        "batch_sizes": [1, 16, 64],
        "warmup_iterations": 1,
        "iterations": 2,
    },
}

PRESET_DESCRIPTIONS = {
    "standard": "LLM-Inference-Bench grid (in=out 128-2048, batch 1-64), 2 warmup + 5 measured",
    "argonne": "Exact LLM-Inference-Bench settings, 10 warmup + 30 measured (slow)",
    "long-context": "v3 grid (1K-256K input, 500 output, batch 1-64), 1 warmup + 3 measured",
    "quick": "Smoke test (in=out 128/512/2048, batch 1/16/64), 1 warmup + 2 measured",
}

DEFAULT_PRESET = "standard"

# Keys that together define the workload grid; a layer that sets one style clears the other.
_PAIRED_KEYS = {"io_lengths"}
_CROSS_KEYS = {"input_lengths", "output_lengths"}


@dataclass
class BenchConfig:
    """All knobs for one benchmark run."""

    # Engine
    engine: str = "endpoint"  # endpoint | offline
    url: str = "http://localhost:8000"
    api: str = "completions"  # completions | chat
    model: Optional[str] = None  # served model id (endpoint) or HF path (offline); auto-detected when unset
    prompt_format: str = "token_ids"  # token_ids | text

    # Workload grid (LLM-Inference-Bench style)
    preset: str = DEFAULT_PRESET
    batch_sizes: List[int] = field(default_factory=lambda: list(PRESETS[DEFAULT_PRESET]["batch_sizes"]))
    io_lengths: Optional[List[int]] = field(default_factory=lambda: list(PRESETS[DEFAULT_PRESET]["io_lengths"]))
    input_lengths: Optional[List[int]] = None
    output_lengths: Optional[List[int]] = None
    warmup_iterations: int = 2
    iterations: int = 5

    # Sampling (LLM-Inference-Bench: temperature=1.0, top_p=1.0, ignore_eos=True)
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = True
    # None = a fresh random seed per run (recorded in the results). A fixed seed reproduces the
    # exact prompts, but re-running it against a live server hits that server's prefix cache.
    seed: Optional[int] = None
    # Argonne draws token ids from [0, 10000); the first 1000 ids are skipped because
    # many tokenizers keep control tokens there.
    token_id_range: Tuple[int, int] = (1000, 10000)

    # Run control
    cooldown_seconds: float = 3.0
    request_timeout: float = 1800.0
    max_seconds_per_config: Optional[float] = None
    skip_exceeding_context: bool = True

    # Labels / metadata overrides
    label: Optional[str] = None  # display name used in titles and file names
    hardware: Optional[str] = None  # override detected GPU description
    framework: Optional[str] = None  # override detected framework (vLLM, SGLang, ...)
    quantization: Optional[str] = None
    num_gpus: Optional[int] = None  # "Num of Hardware" in the LLM-Inference-Bench CSV
    tensor_parallel: Optional[int] = None
    pipeline_parallel: Optional[int] = None

    # Telemetry
    gpu_monitor: bool = True
    gpu_poll_interval: float = 0.1
    gpu_indices: Optional[List[int]] = None  # default: all visible GPUs
    server_metrics: bool = True
    idle_power_seconds: float = 3.0

    # Offline engine (vLLM LLM() in-process, as in LLM-Inference-Bench)
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    enable_prefix_caching: Optional[bool] = None
    engine_args: Dict[str, Any] = field(default_factory=dict)

    # Output
    output_dir: str = "./outputs"
    png_dpi: int = 300

    def shapes(self) -> List[Tuple[int, int]]:
        """(input_len, output_len) pairs to test, in run order."""
        if self.io_lengths:
            return [(n, n) for n in self.io_lengths]
        return [(i, o) for i in self.input_lengths or [] for o in self.output_lengths or []]

    def test_queue(self) -> List[Tuple[int, int, int]]:
        """(input_len, output_len, batch_size) for every test, shapes outermost like v3."""
        return [(i, o, b) for (i, o) in self.shapes() for b in self.batch_sizes]

    @property
    def paired_io(self) -> bool:
        return bool(self.io_lengths)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["token_id_range"] = list(self.token_id_range)
        return data


def _field_names() -> set:
    return {f.name for f in fields(BenchConfig)}


def load_config_file(path: str) -> Dict[str, Any]:
    """Read a TOML or JSON config file; tables/sections are flattened into one namespace."""
    p = Path(path)
    if p.suffix.lower() == ".toml":
        try:
            import tomllib
        except ImportError:  # Python 3.10
            try:
                import tomli as tomllib
            except ImportError as e:
                raise ValueError("TOML configs need Python 3.11+ or `pip install tomli`; or use a .json config") from e

        with open(p, "rb") as f:
            raw = tomllib.load(f)
    else:
        with open(p) as f:
            raw = json.load(f)

    flat: Dict[str, Any] = {}
    known = _field_names()

    def walk(table: Dict[str, Any]) -> None:
        for key, value in table.items():
            key = key.replace("-", "_")
            if isinstance(value, dict) and key not in ("engine_args",):
                walk(value)
            else:
                flat[key] = value

    walk(raw)
    unknown = sorted(k for k in flat if k not in known)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")
    return flat


def _apply_layer(values: Dict[str, Any], layer: Dict[str, Any]) -> None:
    """Merge one layer, keeping the paired (io_lengths) and cross (input x output) styles exclusive."""
    layer = {k: v for k, v in layer.items() if v is not None}
    if _PAIRED_KEYS & layer.keys():
        values.pop("input_lengths", None)
        values.pop("output_lengths", None)
    if _CROSS_KEYS & layer.keys():
        # Setting only one half of a cross grid keeps the previous lengths for the other half,
        # e.g. `--output-lengths 500` on the standard preset tests inputs 128..2048 with 500 out.
        previous = values.get("io_lengths")
        if previous:
            values.setdefault("input_lengths", list(previous))
            values.setdefault("output_lengths", list(previous))
        values["io_lengths"] = None
    values.update(layer)


def build_config(
    preset: Optional[str] = None,
    file_values: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> BenchConfig:
    """Resolve defaults < preset < file < CLI overrides into a validated BenchConfig."""
    file_values = dict(file_values or {})
    overrides = dict(overrides or {})
    preset_name = overrides.get("preset") or file_values.get("preset") or preset or DEFAULT_PRESET
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {', '.join(PRESETS)}")

    values: Dict[str, Any] = {}
    _apply_layer(values, PRESETS[preset_name])
    _apply_layer(values, file_values)
    _apply_layer(values, overrides)
    values["preset"] = preset_name

    # A cross grid needs both halves; fill a missing half from a paired grid if one was set.
    if values.get("io_lengths") is None:
        values.setdefault("input_lengths", None)
        values.setdefault("output_lengths", None)
        if not values.get("input_lengths") or not values.get("output_lengths"):
            raise ValueError("Set both input_lengths and output_lengths (or io_lengths for input = output).")

    if "token_id_range" in values:
        values["token_id_range"] = tuple(values["token_id_range"])

    cfg = BenchConfig(**{k: v for k, v in values.items() if k in _field_names()})
    validate(cfg)
    return cfg


def validate(cfg: BenchConfig) -> None:
    if cfg.engine not in ("endpoint", "offline"):
        raise ValueError("engine must be 'endpoint' or 'offline'")
    if cfg.api not in ("completions", "chat"):
        raise ValueError("api must be 'completions' or 'chat'")
    if cfg.prompt_format not in ("token_ids", "text"):
        raise ValueError("prompt_format must be 'token_ids' or 'text'")
    if cfg.api == "chat" and cfg.prompt_format == "token_ids":
        # Chat endpoints only take text; fall back rather than fail.
        cfg.prompt_format = "text"
    if cfg.engine == "offline":
        cfg.prompt_format = "token_ids"
        if not cfg.model:
            raise ValueError("Offline engine needs a model path or Hugging Face id (--model)")
    if not cfg.batch_sizes or any(int(b) < 1 for b in cfg.batch_sizes):
        raise ValueError("batch_sizes must be positive integers")
    cfg.batch_sizes = sorted({int(b) for b in cfg.batch_sizes})
    for name in ("io_lengths", "input_lengths", "output_lengths"):
        vals = getattr(cfg, name)
        if vals is not None:
            if any(int(v) < 1 for v in vals):
                raise ValueError(f"{name} must be positive integers")
            setattr(cfg, name, sorted({int(v) for v in vals}))
    if cfg.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if cfg.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be >= 0")
    lo, hi = cfg.token_id_range
    if not 0 <= lo < hi:
        raise ValueError("token_id_range must be [low, high) with 0 <= low < high")
    if cfg.seed is None:
        cfg.seed = secrets.randbits(31)
    cfg.url = cfg.url.rstrip("/")


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    """Parse '1,16,32' or '1K,4K,128K' into ints (K = 1000, Ki = 1024)."""
    if text is None:
        return None
    out = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part.endswith("ki"):
            out.append(int(float(part[:-2]) * 1024))
        elif part.endswith("k"):
            out.append(int(float(part[:-1]) * 1000))
        else:
            out.append(int(part))
    return out
