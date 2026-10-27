"""
v3-style PNG report and console summaries for v4 results.

The figure keeps v3's look and row order (throughput landscape, heatmaps, UX metrics,
generation efficiency, GPU metrics) with x = input length and one series per batch size,
and adds what v3 could not show: a KPI strip, LLM-Inference-Bench total throughput and batch
latency, measured TTFT/TPOT, energy per token from the GPU energy counter, power-cap
residency, tail latency, server internals (KV cache, queueing, preemptions, speculative
decoding) and a whole-run telemetry timeline.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
from rich import box
from rich.console import Console
from rich.table import Table

from .util import fmt_duration, fmt_shape, fmt_tokens

# v3 palette first (same look), then extra colors so 7+ batch sizes never repeat a color
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51", "#5E548E", "#3D405B", "#81B29A", "#E07A5F"]
MARKERS_CYCLE = ["o", "s", "^", "D", "v", "p", "h", "*"]
FACE = "#FAFAFA"
ROW_HEIGHT_IN = 5.0
FIG_WIDTH_IN = 24.0


# ----------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------
class ReportData:
    def __init__(self, package: Dict[str, Any]):
        self.meta = package.get("metadata", {})
        self.cfg = self.meta.get("configuration", {})
        self.system = self.meta.get("system_info", {})
        self.server = self.meta.get("server_info", {})
        self.telemetry = self.meta.get("telemetry", {})
        self.all = pd.DataFrame(package.get("results", []))
        if self.all.empty:
            raise ValueError("No results to plot")
        self.ok = self.all[self.all["status"].isin(["ok", "partial"])].copy()
        self.paired = bool(self.cfg.get("io_lengths"))
        outputs = sorted(self.all["output_len"].unique())
        if self.paired or len(outputs) == 1:
            self.ref_output = None if self.paired else int(outputs[0])
            self.main = self.ok.copy()
            self.main_all = self.all.copy()
        else:
            counts = self.ok["output_len"].value_counts() if not self.ok.empty else self.all["output_len"].value_counts()
            top = counts[counts == counts.max()].index
            self.ref_output = int(max(top))
            self.main = self.ok[self.ok["output_len"] == self.ref_output].copy()
            self.main_all = self.all[self.all["output_len"] == self.ref_output].copy()
        # Line charts only get ticks where at least one test succeeded; heatmaps show every row
        source = self.main if not self.main.empty else self.main_all
        self.inputs = sorted(int(x) for x in source["input_len"].unique())
        self.batches = sorted(int(x) for x in self.all["batch_size"].unique())
        self.cross = (not self.paired) and len(outputs) >= 2 and self.all["input_len"].nunique() >= 2
        self.label = self.meta.get("label") or self.meta.get("model") or "model"

    def has(self, col: str, df: Optional[pd.DataFrame] = None) -> bool:
        df = self.main if df is None else df
        return col in df.columns and df[col].notna().any()

    @property
    def x_label(self) -> str:
        if self.paired:
            return "Input = Output Length (tokens)"
        return f"Input Length (tokens), output = {fmt_tokens(self.ref_output)}"

    @property
    def y_label(self) -> str:
        return "Input = Output" if self.paired else "Input Length"

    def color(self, batch: int) -> str:
        return COLORS[self.batches.index(batch) % len(COLORS)]

    def config_str(self, row: pd.Series) -> str:
        return f"{fmt_shape(row['input_len'], row['output_len'], self.paired)} × B{int(row['batch_size'])}"


# ----------------------------------------------------------------------------------------
# Panel helpers (v3 styling)
# ----------------------------------------------------------------------------------------
def _style_line_axes(ax, data: ReportData, ylabel: str, title: str, big: bool = False) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(data.inputs)
    ax.set_xticklabels([fmt_tokens(x) for x in data.inputs])
    ax.minorticks_off()
    ax.set_xlabel(data.x_label, fontsize=13 if big else 11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13 if big else 11, fontweight="bold")
    ax.set_title(title, fontsize=15 if big else 12, fontweight="bold", pad=15 if big else 10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor(FACE)


def _na(ax, title: str, msg: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#777777", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(FACE)


def _series(
    ax,
    data: ReportData,
    col: str,
    ylabel: str,
    title: str,
    marker: str = "o",
    scale: float = 1.0,
    band: Optional[Tuple[str, str]] = None,
    legend: bool = False,
    big: bool = False,
    skip_batches: Tuple[int, ...] = (),
) -> bool:
    if not data.has(col):
        _na(ax, title, "Not available for this run")
        return False
    for b in data.batches:
        if b in skip_batches:
            continue
        d = data.main[(data.main["batch_size"] == b) & data.main[col].notna()].sort_values("input_len")
        if d.empty:
            continue
        color = data.color(b)
        ax.plot(
            d["input_len"],
            d[col] * scale,
            marker=marker,
            linewidth=3 if big else 2.5,
            markersize=10 if big else 9,
            label=f"batch {b}",
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        if band and band[0] in d.columns and band[1] in d.columns:
            lo, hi = d[band[0]], d[band[1]]
            if lo.notna().all() and hi.notna().all():
                ax.fill_between(d["input_len"], lo * scale, hi * scale, color=color, alpha=0.12, linewidth=0)
    _style_line_axes(ax, data, ylabel, title, big=big)
    if legend:
        if big:
            ax.legend(title="Batch Size", fontsize=10, title_fontsize=11, loc="best", frameon=True, shadow=True)
        else:
            ax.legend(fontsize=9, loc="best")
    return True


def _ux_bands(ax, bands: List[Tuple[float, float, str]], alpha: float = 0.08) -> None:
    """v3's green/yellow/orange UX quality bands, without letting them rescale the axis."""
    ymin, ymax = ax.get_ylim()
    for lo, hi, color in bands:
        if lo < ymax:
            ax.axhspan(lo, hi, alpha=alpha, color=color, zorder=0)
    ax.set_ylim(max(0.0, ymin), ymax)


def _heatmap(
    ax,
    data: ReportData,
    col: str,
    title: str,
    cbar_label: str,
    fmt: str,
    cmap: str,
    scale: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    df_ok: Optional[pd.DataFrame] = None,
    df_all: Optional[pd.DataFrame] = None,
    index: str = "input_len",
    columns: str = "batch_size",
    xlabel: str = "Batch Size",
    ylabel: Optional[str] = None,
) -> bool:
    df_ok = data.main if df_ok is None else df_ok
    df_all = data.main_all if df_all is None else df_all
    if col not in df_ok.columns or df_ok[col].isna().all():
        _na(ax, title, "Not available for this run")
        return False
    rows = sorted(int(x) for x in df_all[index].unique())
    cols = sorted(int(x) for x in df_all[columns].unique())
    pivot = (
        df_ok.pivot_table(index=index, columns=columns, values=col, aggfunc="mean").reindex(index=rows, columns=cols)
        * scale
    )
    many = len(cols) > 7
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": cbar_label, "shrink": 0.9},
        linewidths=1.5,
        linecolor="white",
        annot_kws={"fontsize": 8 if many else 10, "weight": "bold"},
    )
    # Mark tests that failed (e.g. OOM) or were skipped (longer than max_model_len)
    status = {(int(r[index]), int(r[columns])): r["status"] for _, r in df_all.iterrows()}
    for (ri, ci), st in status.items():
        if st in ("failed", "skipped") and ri in rows and ci in cols:
            y, x = rows.index(ri), cols.index(ci)
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#E6E6E6", edgecolor="white", lw=1.5, hatch="//", zorder=2))
            ax.text(x + 0.5, y + 0.5, "FAIL" if st == "failed" else "SKIP", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="#8A1C1C" if st == "failed" else "#666666", zorder=3)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel or data.y_label, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_yticklabels([fmt_tokens(y) for y in rows], rotation=0, fontsize=9)
    if columns != "batch_size":
        ax.set_xticklabels([fmt_tokens(c) for c in cols], rotation=0, fontsize=9)
    return True


# ----------------------------------------------------------------------------------------
# Individual panels
# ----------------------------------------------------------------------------------------
def panel_kpis(ax, data: ReportData) -> None:
    ax.axis("off")
    ok = data.ok
    cards: List[Tuple[str, str, str]] = []

    def best(col: str, fn: str = "max"):
        if col not in ok.columns or ok[col].isna().all():
            return None
        return ok.loc[ok[col].idxmax() if fn == "max" else ok[col].idxmin()]

    r = best("output_throughput")
    if r is not None:
        cards.append(("Peak Generation Throughput", f"{r['output_throughput']:,.0f} tok/s", data.config_str(r)))
    r = best("throughput")
    if r is not None:
        cards.append(("Peak Total Throughput", f"{r['throughput']:,.0f} tok/s", f"input+output · {data.config_str(r)}"))
    if data.has("decode_speed_per_user", ok):
        smallest = ok[ok["batch_size"] == ok["batch_size"].min()]
        r = smallest.loc[smallest["decode_speed_per_user"].idxmax()] if smallest["decode_speed_per_user"].notna().any() else None
        if r is not None:
            cards.append(("Single-Stream Decode", f"{r['decode_speed_per_user']:,.1f} tok/s", data.config_str(r)))
    r = best("ttft_p50_ms", "min")
    if r is not None:
        cards.append(("Best TTFT (p50)", f"{r['ttft_p50_ms']:,.0f} ms", data.config_str(r)))
    r = best("tokens_per_joule")
    if r is not None:
        cards.append(("Best Energy Efficiency", f"{r['tokens_per_joule']:.2f} tok/J", f"{r['energy_per_output_token']:.3f} J/tok · {data.config_str(r)}"))
    n_ok = int(data.all["status"].isin(["ok", "partial"]).sum())
    summary = f"{n_ok}/{len(data.all)} tests ok"
    if "energy_watt_hours" in ok.columns and ok["energy_watt_hours"].notna().any():
        summary += f" · {ok['energy_watt_hours'].sum():.1f} Wh measured"
    cards.append(("Run", fmt_duration(data.meta.get("benchmark_duration")), summary))

    n = len(cards)
    gap = 0.012
    width = (1 - gap * (n - 1)) / n
    for k, (title, value, sub) in enumerate(cards):
        x = k * (width + gap)
        color = COLORS[k % len(COLORS)]
        ax.add_patch(FancyBboxPatch((x, 0.04), width, 0.92, boxstyle="round,pad=0.0,rounding_size=0.04",
                                    transform=ax.transAxes, facecolor="white", edgecolor="#DDDDDD", linewidth=1.5))
        ax.add_patch(Rectangle((x, 0.04), 0.006, 0.92, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        cx = x + width / 2
        ax.text(cx, 0.78, title, ha="center", va="center", fontsize=11, color="#555555", fontweight="bold", transform=ax.transAxes)
        ax.text(cx, 0.47, value, ha="center", va="center", fontsize=21, color=color, fontweight="bold", transform=ax.transAxes)
        ax.text(cx, 0.18, sub, ha="center", va="center", fontsize=10, color="#444444", transform=ax.transAxes)


def panel_throughput_landscape(ax, data: ReportData) -> None:
    _series(ax, data, "output_throughput", "Generation Throughput (tokens/sec)",
            "Generation Throughput vs Sequence Length by Batch Size (band: p10-p90 across iterations)",
            band=("output_throughput_p10", "output_throughput_p90"), legend=True, big=True)
    if data.has("output_throughput"):
        top = data.main.loc[data.main["output_throughput"].idxmax()]
        ax.annotate(f"peak {top['output_throughput']:,.0f} tok/s", (top["input_len"], top["output_throughput"]),
                    xytext=(0, 14), textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color="#333333")


def panel_power(ax, data: ReportData) -> None:
    if not _series(ax, data, "avg_power", "Power (W)", "Average Power Draw (all monitored GPUs)", marker="o"):
        return
    ax.yaxis.label.set_color("#E63946")
    ax.tick_params(axis="y", labelcolor="#E63946")
    limits = [x for x in data.system.get("power_limits_w") or [] if x]
    if limits:
        ax.axhline(sum(limits), color="#E63946", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(0.01, sum(limits), f" power limit {sum(limits):.0f}W", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=9, color="#E63946")
    idle = data.telemetry.get("idle_power_w")
    if idle:
        ax.axhline(idle, color="#777777", linestyle=":", linewidth=1.5)
        ax.text(0.01, idle, f" idle {idle:.0f}W", transform=ax.get_yaxis_transform(), va="bottom", fontsize=9, color="#555555")
    ax.set_ylim(bottom=0)


def panel_clock(ax, data: ReportData) -> None:
    if not _series(ax, data, "avg_gpu_clock", "Clock (MHz)", "GPU Clock Frequency (avg SM clock)", marker="H"):
        return
    max_clocks = [x for x in data.system.get("max_sm_clock_mhz") or [] if x]
    if max_clocks:
        m = float(np.mean(max_clocks))
        ax.axhline(m, color="#2E86AB", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(0.01, m, f" max {m:.0f} MHz", transform=ax.get_yaxis_transform(), va="bottom", fontsize=9, color="#2E86AB")


def panel_timeline(ax, data: ReportData) -> None:
    tl = data.meta.get("timeline") or {}
    windows = data.meta.get("windows") or []
    title = "Run Timeline: GPU Power and SM Clock (shaded = measured windows, colored by batch size)"
    if not tl.get("t"):
        _na(ax, title, "No GPU telemetry recorded")
        return
    t = np.array(tl["t"], dtype=float) / 60.0
    power = np.array([np.nan if v is None else v for v in tl.get("power_draw", [])], dtype=float)
    clock = np.array([np.nan if v is None else v for v in tl.get("gpu_clock", [])], dtype=float)
    shapes_seen = set()
    t_end = float(t[-1]) if len(t) else 1.0
    last_label_x = -1e9
    for w in windows:
        b = w["batch_size"]
        color = data.color(b) if b in data.batches else "#999999"
        ax.axvspan(w["test_start"] / 60, w["measure_start"] / 60, color="#BBBBBB", alpha=0.15, linewidth=0)
        ax.axvspan(w["measure_start"] / 60, w["measure_end"] / 60, color=color, alpha=0.18, linewidth=0)
        key = (w["input_len"], w["output_len"])
        if key not in shapes_seen:
            shapes_seen.add(key)
            x = w["test_start"] / 60
            ax.axvline(x, color="#666666", linewidth=0.8, alpha=0.6)
            if x - last_label_x > 0.045 * t_end:  # keep shape labels from overlapping
                ax.text(x, 1.01, fmt_shape(key[0], key[1], data.paired), transform=ax.get_xaxis_transform(),
                        fontsize=9, fontweight="bold", color="#444444", ha="left", va="bottom")
                last_label_x = x
    ax.plot(t, power, color="#E63946", linewidth=1.4, label="Power (W)")
    limits = [x for x in data.system.get("power_limits_w") or [] if x]
    if limits:
        ax.axhline(sum(limits), color="#E63946", linestyle="--", linewidth=1.2, alpha=0.7, label="Power limit")
    ax.set_ylabel("Power (W)", fontsize=13, fontweight="bold", color="#E63946")
    ax.tick_params(axis="y", labelcolor="#E63946")
    ax.set_xlabel("Elapsed (minutes)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=24)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor(FACE)
    ax.set_xlim(0, max(t) if len(t) else 1)
    ax.set_ylim(bottom=0)
    if np.isfinite(clock).any():
        ax2 = ax.twinx()
        ax2.plot(t, clock, color="#2E86AB", linewidth=1.1, alpha=0.85, label="SM clock (MHz)")
        ax2.set_ylabel("SM Clock (MHz)", fontsize=13, fontweight="bold", color="#2E86AB")
        ax2.tick_params(axis="y", labelcolor="#2E86AB")
        ax2.set_ylim(bottom=0)
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=10, frameon=True)


# ----------------------------------------------------------------------------------------
# Figure assembly
# ----------------------------------------------------------------------------------------
Panel = Tuple[str, Callable]  # (span: "full" | "two" | "one", draw(ax))


def _build_rows(data: ReportData) -> List[Tuple[float, List[Panel]]]:
    rows: List[Tuple[float, List[Panel]]] = []
    m = data.main

    rows.append((0.42, [("full", lambda ax: panel_kpis(ax, data))]))
    rows.append((1.2, [("full", lambda ax: panel_throughput_landscape(ax, data))]))
    rows.append((1.0, [
        ("two", lambda ax: _heatmap(ax, data, "throughput", "Total Throughput Heatmap (LLM-Inference-Bench: B×(in+out)/latency)",
                                    "Tokens/sec", ".0f", "RdYlGn")),
        ("one", lambda ax: _heatmap(ax, data, "latency_avg", "Batch Latency Heatmap (seconds)", "Latency (sec)", ".1f", "RdYlGn_r")),
    ]))

    def ttft(ax):
        probed = "ttft_source" in m.columns and (m["ttft_source"] == "probe").any()
        source = "max_tokens=1 probe batch" if probed else "p50, measured"
        if _series(ax, data, "ttft_p50_ms", "TTFT (milliseconds)", f"Time to First Token ({source})", marker="*"):
            _ux_bands(ax, [(0, 200, "green"), (200, 1000, "yellow"), (1000, 3000, "orange")])

    def e2e(ax):
        if _series(ax, data, "e2e_mean", "Latency (seconds)", "Request Latency (end-to-end mean)", marker="D"):
            _ux_bands(ax, [(0, 2, "green"), (2, 5, "yellow"), (5, 10, "orange")])

    rows.append((1.0, [
        ("one", lambda ax: _series(ax, data, "throughput_per_user", "Tokens/sec per Request", "Throughput per User", marker="s", legend=True)),
        ("one", ttft),
        ("one", e2e),
    ]))

    def itl(ax):
        if _series(ax, data, "tpot_mean_ms", "ms per output token", "Inter-Token Latency (TPOT, measured)", marker="v"):
            _ux_bands(ax, [(0, 50, "green"), (50, 100, "yellow"), (100, 200, "orange")], alpha=0.1)

    rows.append((1.0, [
        ("one", lambda ax: _series(ax, data, "prefill_throughput", "Prefill Speed (tokens/sec)",
                                   "Prompt Processing Speed (batch prompt tokens / last TTFT)", marker="^")),
        ("one", itl),
        ("one", lambda ax: _series(ax, data, "decode_speed_per_user", "Tokens/sec per Request", "Decode Speed (1 / TPOT)", marker="o")),
    ]))

    if data.has("avg_power") or data.has("avg_gpu_clock"):
        rows.append((1.0, [
            ("one", lambda ax: panel_power(ax, data)),
            ("one", lambda ax: panel_clock(ax, data)),
            ("one", lambda ax: _series(ax, data, "batch_efficiency", "Efficiency (%)",
                                       "Batch Scaling Efficiency (vs smallest batch)", marker="p",
                                       skip_batches=(min(data.batches),))),
        ]))

    if data.has("energy_per_output_token"):
        if data.has("power_cap_pct"):
            third = ("one", lambda ax: _heatmap(ax, data, "power_cap_pct", "Power-Cap Throttling (% of samples)",
                                                "% of time power-capped", ".0f", "Reds", vmin=0, vmax=100))
        else:
            third = ("one", lambda ax: _heatmap(ax, data, "max_temperature", "Peak GPU Temperature (C)", "Temperature (C)",
                                                ".0f", "YlOrRd"))
        rows.append((1.0, [
            ("one", lambda ax: _series(ax, data, "energy_per_output_token", "Joules per generated token",
                                       "Energy per Output Token (GPU energy counter)", marker="o")),
            ("one", lambda ax: _heatmap(ax, data, "tokens_per_joule", "Energy Efficiency (generated tokens / joule)",
                                        "tok/J", ".2f", "RdYlGn")),
            third,
        ]))

    tail = []
    if data.has("ttft_p99_ms"):
        tail.append(("one", lambda ax: _heatmap(ax, data, "ttft_p99_ms", "TTFT p99 (ms)", "ms", ".0f", "RdYlGn_r")))
    if data.has("itl_p99_ms"):
        tail.append(("one", lambda ax: _heatmap(ax, data, "itl_p99_ms", "Stream Stall: ITL p99 (ms between chunks)", "ms", ".0f", "RdYlGn_r")))
    if data.has("latency_cv") and data.main["iterations"].max() > 1:
        tail.append(("one", lambda ax: _heatmap(ax, data, "latency_cv", "Run-to-Run Variability (batch latency CV %)",
                                                "CV (%)", ".1f", "RdYlGn_r", vmin=0)))
    if tail:
        rows.append((1.0, tail))

    server_panels: List[Panel] = []
    if data.has("kv_cache_usage_peak"):
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "kv_cache_usage_peak", "Peak KV Cache Usage (%)",
                                                         "% of KV cache", ".0f", "RdYlGn_r", vmin=0, vmax=100)))
    if data.has("server_queue_time_mean"):
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "server_queue_time_mean", "Server Queue Time (mean ms)",
                                                         "ms waiting before scheduling", ".0f", "RdYlGn_r", scale=1000)))
    elif data.has("waiting_peak"):
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "waiting_peak", "Peak Waiting Requests",
                                                         "requests", ".0f", "RdYlGn_r")))
    if data.has("preemptions"):
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "preemptions", "Preemptions (KV cache pressure)",
                                                         "count", ".0f", "RdYlGn_r", vmin=0)))
    if data.has("spec_mean_acceptance_length"):
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "spec_mean_acceptance_length",
                                                         "Speculative Decoding: Mean Acceptance Length", "tokens/step", ".2f", "RdYlGn")))
    if data.has("prefix_cache_hit_rate") and data.main["prefix_cache_hit_rate"].max() > 1.0:
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "prefix_cache_hit_rate", "Prefix Cache Hit Rate (%)",
                                                         "Hit Rate (%)", ".1f", "RdYlGn", vmin=0, vmax=100)))
    if data.has("tokens_per_chunk") and data.main["tokens_per_chunk"].max() > 1.05:
        server_panels.append(("one", lambda ax: _heatmap(ax, data, "tokens_per_chunk", "Tokens per Streamed Chunk",
                                                         "tokens", ".2f", "Blues")))
    for k in range(0, len(server_panels), 3):
        rows.append((1.0, server_panels[k : k + 3]))

    if (data.meta.get("timeline") or {}).get("t"):
        rows.append((1.0, [("full", lambda ax: panel_timeline(ax, data))]))

    if data.cross:
        big_b = max(data.batches)
        small_b = min(data.batches)
        sub_big = data.ok[data.ok["batch_size"] == big_b]
        sub_small = data.ok[data.ok["batch_size"] == small_b]
        all_big = data.all[data.all["batch_size"] == big_b]
        all_small = data.all[data.all["batch_size"] == small_b]
        rows.append((1.0, [
            ("two", lambda ax: _heatmap(ax, data, "output_throughput", f"Input × Output: Generation Throughput (batch {big_b})",
                                        "Tokens/sec", ".0f", "RdYlGn", df_ok=sub_big, df_all=all_big,
                                        columns="output_len", xlabel="Output Length", ylabel="Input Length")),
            ("one", lambda ax: _heatmap(ax, data, "output_throughput", f"Input × Output (batch {small_b})",
                                        "Tokens/sec", ".0f", "RdYlGn", df_ok=sub_small, df_all=all_small,
                                        columns="output_len", xlabel="Output Length", ylabel="Input Length")),
        ]))
    return rows


def _subtitle(data: ReportData) -> List[str]:
    s, v, cfg, meta = data.system, data.server, data.cfg, data.meta
    line1 = []
    gpu = s.get("gpu_description") or s.get("gpu_name")
    if gpu:
        vram = s.get("total_vram_gb")
        line1.append(f"GPU: {gpu}" + (f" ({vram:.0f}GB)" if vram and not s.get("gpu_description") else ""))
    limits = [x for x in s.get("power_limits_w") or [] if x]
    if limits:
        uniq = sorted(set(round(x) for x in limits))
        line1.append("Power limit: " + ("/".join(f"{u}W" for u in uniq)) + (f" × {len(limits)}" if len(limits) > 1 else ""))
    if s.get("driver_version"):
        line1.append(f"Driver {s['driver_version']}")
    if s.get("cuda_version"):
        line1.append(f"CUDA {s['cuda_version']}")

    line2 = []
    fw = " ".join(str(x) for x in (v.get("framework"), v.get("version")) if x)
    if fw:
        line2.append(fw)
    if v.get("model_name") and v.get("model_name") != data.label:
        line2.append(f"Model: {v['model_name']}")
    if v.get("quantization"):
        line2.append(f"Quant: {v['quantization']}")
    par = []
    if v.get("tensor_parallel"):
        par.append(f"TP={v['tensor_parallel']}")
    if v.get("pipeline_parallel") and v.get("pipeline_parallel") != 1:
        par.append(f"PP={v['pipeline_parallel']}")
    if par:
        line2.append(", ".join(par))
    if v.get("max_model_len"):
        line2.append(f"Max ctx {fmt_tokens(v['max_model_len'])}")
    if v.get("kv_cache_capacity_tokens"):
        line2.append(f"KV cache {v['kv_cache_capacity_tokens']:,} tok")
    if v.get("prefix_caching") is not None:
        line2.append("Prefix cache " + ("on" if v["prefix_caching"] else "off"))
    if v.get("speculative_decoding") or v.get("speculative_algorithm"):
        line2.append("Spec decode" + (f" ({v['speculative_algorithm']})" if v.get("speculative_algorithm") else ""))

    prompts = "random token-id prompts" if cfg.get("prompt_format") == "token_ids" else "random-word prompts"
    line3 = [
        f"Method: LLM-Inference-Bench ({prompts}, ignore_eos, temp {cfg.get('temperature')}, top-p {cfg.get('top_p')})",
        f"{cfg.get('warmup_iterations')} warmup + {cfg.get('iterations')} measured batches/test",
    ]
    if v.get("api"):
        line3.append(f"{v['api']} (streaming)" if v.get("endpoint") != "in-process" else "offline LLM.generate")
    if data.telemetry.get("idle_power_w"):
        line3.append(f"Idle {data.telemetry['idle_power_w']:.0f}W")
    started = meta.get("started_at")
    try:
        ts = datetime.fromisoformat(started).strftime("%Y-%m-%d %H:%M") if started else None
    except ValueError:
        ts = None
    line3.append(f"Timestamp: {ts or datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines = [" | ".join(line1), " | ".join(line2), " | ".join(line3)]
    return [l for l in lines if l]


def render_report(package: Dict[str, Any], png_path, dpi: int = 300) -> Path:
    data = ReportData(package)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "hatch.color": "#BBBBBB",
        }
    )
    rows = _build_rows(data)
    header_in = 2.3
    footer_in = 1.1
    heights = [h * ROW_HEIGHT_IN for h, _ in rows]
    gap_in = 1.25  # vertical space between rows for titles/labels
    fig_h = header_in + footer_in + sum(heights) + gap_in * (len(rows) - 1)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))
    gs = fig.add_gridspec(
        len(rows), 3,
        hspace=gap_in / (sum(heights) / len(rows)),
        wspace=0.32,
        left=0.06, right=0.98,
        top=1 - header_in / fig_h,
        bottom=footer_in / fig_h,
        height_ratios=[h for h, _ in rows],
    )
    for r, (_, panels) in enumerate(rows):
        col = 0
        for span, draw in panels:
            if span == "full":
                ax = fig.add_subplot(gs[r, :])
            elif span == "two":
                ax = fig.add_subplot(gs[r, col : col + 2])
                col += 2
            else:
                ax = fig.add_subplot(gs[r, col])
                col += 1
            draw(ax)

    # Header (v3 layout: bold title + wheat subtitle box)
    fig.suptitle(f"{data.label} - Performance Report", fontsize=18, fontweight="bold", y=1 - 0.35 / fig_h)
    subtitle = "\n".join(_subtitle(data))
    fig.text(0.5, 1 - 1.45 / fig_h, subtitle, ha="center", va="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3))
    meta = data.meta
    if not meta.get("complete", True):
        reason = meta.get("aborted") or "run did not finish"
        fig.text(0.5, 1 - 2.05 / fig_h, f"INCOMPLETE RUN: {reason}", ha="center", va="center", fontsize=12,
                 fontweight="bold", color="#B00020")
    fail = data.all[data.all["status"].isin(["failed", "skipped"])]
    footer = (
        "Throughput = B×(input+output)/batch latency and Latency = mean batch wall time, as defined by LLM-Inference-Bench "
        "(Chitty-Venkata et al., arXiv:2411.00136). Generation throughput = generated tokens/sec (v3's tokens/sec). "
        "TTFT/TPOT/ITL measured from the token stream."
    )
    if not fail.empty:
        footer += f"  FAIL/SKIP cells: {len(fail)} test(s) failed or exceeded max_model_len."
    fig.text(0.5, 0.2 / fig_h, footer, ha="center", va="center", fontsize=10, color="#555555", wrap=True)

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


# ----------------------------------------------------------------------------------------
# Console summaries
# ----------------------------------------------------------------------------------------
def print_highlights(console: Console, package: Dict[str, Any]) -> None:
    data = ReportData(package)
    ok = data.ok
    if ok.empty:
        console.print("[red]No successful tests.[/red]")
        return

    def add(table, label, col, fmt, fn="max"):
        if col not in ok.columns or ok[col].isna().all():
            return
        r = ok.loc[ok[col].idxmax() if fn == "max" else ok[col].idxmin()]
        table.add_row(label, fmt.format(r[col]), data.config_str(r))

    t = Table(show_header=True, header_style="bold magenta", box=box.DOUBLE, title="Performance Highlights")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="yellow", justify="right")
    t.add_column("Configuration", style="green")
    add(t, "Peak Generation Throughput", "output_throughput", "{:,.1f} tok/s")
    add(t, "Peak Total Throughput (LLM-Inf-Bench)", "throughput", "{:,.1f} tok/s")
    add(t, "Best Efficiency", "throughput_per_user", "{:,.1f} tok/s/user")
    add(t, "Lowest Batch Latency", "latency_avg", "{:.2f}s", "min")
    add(t, "Best TTFT (p50)", "ttft_p50_ms", "{:,.0f} ms", "min")
    add(t, "Fastest Decode (1/TPOT)", "decode_speed_per_user", "{:,.1f} tok/s")
    add(t, "Peak GPU Util", "avg_gpu_util", "{:.1f}%")
    console.print(t)

    if "tokens_per_joule" in ok.columns and ok["tokens_per_joule"].notna().any():
        e = Table(show_header=True, header_style="bold green", box=box.DOUBLE, title="Energy Efficiency Analysis")
        e.add_column("Metric", style="cyan")
        e.add_column("Value", style="yellow", justify="right")
        e.add_column("Configuration", style="green")
        add(e, "Best Energy Efficiency", "tokens_per_joule", "{:.3f} tok/J")
        add(e, "Lowest Energy per Token", "energy_per_output_token", "{:.4f} J/tok", "min")
        add(e, "Best Throughput/Watt (LLM-Inf-Bench)", "throughput_per_watt", "{:.2f} tok/s/W")
        add(e, "Most Power-Capped", "power_cap_pct", "{:.0f}% of samples")
        total_wh = ok["energy_watt_hours"].sum() if "energy_watt_hours" in ok.columns else 0
        e.add_row("Total Energy (measured windows)", f"{total_wh:.2f} Wh", "All tests combined")
        idle = data.telemetry.get("idle_power_w")
        if idle:
            e.add_row("Idle Power", f"{idle:.1f} W", "Before warmup")
        console.print(e)


def print_detailed_summary(console: Console, package: Dict[str, Any]) -> None:
    data = ReportData(package)
    df = data.all
    if console.width < 170:
        # Wide table like v3's 180-column summary; let narrow terminals wrap instead of truncating
        console = Console(file=console.file, width=170, force_terminal=console.is_terminal)
    for (i, o), group in df.groupby(["input_len", "output_len"], sort=True):
        t = Table(title=f"Input {fmt_tokens(i)} / Output {fmt_tokens(o)} tokens", box=box.SIMPLE_HEAVY,
                  header_style="bold magenta", title_style="bold cyan")
        for col in ("Batch", "Latency(s)", "Total tok/s", "Gen tok/s", "Req/s", "TTFT p50", "TTFT p99", "TPOT ms",
                    "ITL p99", "GPU%", "Power W", "J/tok", "KV%", "Status"):
            t.add_column(col, justify="right")

        def f(r, col, fmt):
            v = r.get(col)
            return fmt.format(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "-"

        for _, r in group.sort_values("batch_size").iterrows():
            status = r.get("status")
            style = {"ok": "green", "partial": "yellow"}.get(status, "red")
            t.add_row(
                str(int(r["batch_size"])),
                f(r, "latency_avg", "{:.2f}"),
                f(r, "throughput", "{:,.0f}"),
                f(r, "output_throughput", "{:,.0f}"),
                f(r, "requests_per_second", "{:.2f}"),
                f(r, "ttft_p50_ms", "{:,.0f}"),
                f(r, "ttft_p99_ms", "{:,.0f}"),
                f(r, "tpot_mean_ms", "{:.1f}"),
                f(r, "itl_p99_ms", "{:.0f}"),
                f(r, "avg_gpu_util", "{:.0f}"),
                f(r, "avg_power", "{:.0f}"),
                f(r, "energy_per_output_token", "{:.3f}"),
                f(r, "kv_cache_usage_peak", "{:.0f}"),
                f"[{style}]{status}[/{style}]",
            )
        console.print(t)
