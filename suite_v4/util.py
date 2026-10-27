"""Small formatting helpers shared by the dashboard, console summaries and the PNG report."""

from __future__ import annotations

import re
from typing import Optional


def fmt_tokens(n: Optional[float]) -> str:
    """128 -> '128', 2048 -> '2K', 131072 -> '128K', 48000 -> '48K'."""
    if n is None:
        return "-"
    n = int(n)
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}K"
    if n >= 10000:
        return f"{n / 1000:.0f}K"
    return str(n)


def fmt_shape(input_len: int, output_len: int, paired: bool) -> str:
    return fmt_tokens(input_len) if paired else f"{fmt_tokens(input_len)}→{fmt_tokens(output_len)}"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def sanitize_filename(name: str) -> str:
    """Filesystem-safe name (same rules as v3)."""
    safe = name.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^\w\-.]", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    return safe[:100]
