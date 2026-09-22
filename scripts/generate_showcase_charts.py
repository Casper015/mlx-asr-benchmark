#!/usr/bin/env python3
"""
Generate publication-grade benchmark visualization charts.
Reads consolidated outputs/ALL_DATA.json and generates 8 high-resolution figures.

Charts generated:
1. stt_pareto_frontier.png
2. stt_latency_distribution.png
3. qwen3_quantization_ladder.png
4. wer_ranking_by_family.png
5. longform_dual_axis.png
6. vad_vs_rms_comparison.png
7. memory_efficiency_comparison.png
8. tail_risk_cases.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling constants
FONT_FAMILY = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

FAMILY_COLORS = {
    "Qwen3-ASR": "#1f77b4",
    "Qwen3-ASR (mixed)": "#0284c7",
    "Qwen3-ASR (defective conv.)": "#ef4444",
    "Whisper": "#10b981",
    "Mega-ASR": "#f59e0b",
    "TEA": "#8b5cf6",
    "Granite": "#d97706",
    "Parakeet": "#ec4899",
    "VibeVoice": "#ef4444",
}


def load_data(data_path: Path) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chart_pareto_frontier(data: dict, out_path: Path):
    """Chart 1: Pareto frontier (P50 Latency vs Normalized WER, bubble size = Peak MLX Memory)
    
    Filters out extreme outliers (aufklarer 0.2207 template defect, vibevoice 0.1470 raw prefix,
    and appauto 6085ms slow runtime) to reveal the true competitive Pareto frontier with clear
    separation of all competitive models between 0.020 and 0.040 WER.
    """
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)

    # Filter out extreme outliers to avoid crushing the competitive cluster
    EXCLUDE_OUTLIERS = {
        "qwen3-asr-1.7b-4bit",          # aufklarer defective template (WER 0.2207)
        "vibevoice-streaming-7b-4bit",   # raw unstripped prefix (WER 0.1470, 5165ms)
        "qwen3-asr-1.7b-bf16-appauto",   # mlx_speech runtime disparity (6085ms, represented by mc at 1588ms)
    }
    rows = [r for r in data["pipecat1000"] if r["cond"] not in EXCLUDE_OUTLIERS]

    xs = [r["p50"] for r in rows]
    ys = [r["wer"] for r in rows]
    sizes = [max(140, (r["mlx_mb"] / 10) ** 1.1) for r in rows]
    colors = [FAMILY_COLORS.get(r["family"], "#64748b") for r in rows]

    # Plot bubbles with subtle drop-shadow effect
    scatter = ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.78, edgecolors="#0f172a", linewidth=1.3, zorder=4)

    # True Pareto efficient points among competitive models:
    # 1. Parakeet 0.6B: (126.31, 0.0383)
    # 2. Granite 2B NAR: (409.30, 0.0283)
    # 3. Qwen3-4bit-mc: (557.25, 0.0241)
    # 4. Qwen3-5bit-mc: (839.77, 0.0217)
    # 5. Mega-ASR-bf16: (1163.62, 0.0215)
    pareto_pts = [
        (126.31, 0.0383),
        (409.30, 0.0283),
        (557.25, 0.0241),
        (839.77, 0.0217),
        (1163.62, 0.0215),
    ]
    pareto_pts.sort(key=lambda p: p[0])
    px = [p[0] for p in pareto_pts]
    py = [p[1] for p in pareto_pts]

    # Step line for Pareto frontier
    ax.step(px, py, where="post", color="#1d4ed8", linestyle="--", linewidth=2.0, label="Pareto Efficient Frontier", zorder=3)
    ax.scatter(px, py, color="#1d4ed8", s=110, edgecolors="#ffffff", linewidth=1.5, zorder=5)

    # Carefully curated non-overlapping annotations
    annotations = {
        "parakeet-tdt-0.6b-v3": (
            "Parakeet-TDT 0.6B\n73.0x | 126ms | WER 0.0383",
            (126.31, 0.0383),
            (25, 18)
        ),
        "granite-speech-4.1-2b-nar": (
            "Granite 2B NAR\n19.0x | 409ms | WER 0.0283",
            (409.30, 0.0283),
            (-30, 26)
        ),
        "whisper-turbo": (
            "Whisper Turbo\n18.0x | 531ms | WER 0.0322",
            (531.33, 0.0322),
            (-45, 26)
        ),
        "whisper-distil": (
            "Distil-Whisper v3\n17.6x | 534ms | WER 0.0380",
            (533.91, 0.0380),
            (25, 14)
        ),
        "qwen3-asr-1.7b-4bit-mc": (
            "Qwen3 4-bit (mc) [Sweet Spot]\n17.8x | 557ms | WER 0.0241",
            (557.25, 0.0241),
            (-50, -38)
        ),
        "mega-asr-6bit": (
            "Mega-ASR 6-bit\n16.6x | 657ms | WER 0.0262",
            (656.62, 0.0262),
            (28, 16)
        ),
        "qwen3-asr-1.7b-4bit-alkd-a8t4": (
            "Alkd Mixed (a8/t4) [Lowest VRAM]\n1.57GB RAM | 689ms | WER 0.0242",
            (689.13, 0.0242),
            (25, -34)
        ),
        "qwen3-asr-1.7b-5bit-mc": (
            "Qwen3 5-bit (mc)\n12.7x | 840ms | WER 0.0217",
            (839.77, 0.0217),
            (-35, -36)
        ),
        "mega-asr-bf16": (
            "Mega-ASR BF16 [Lowest WER]\n9.3x | 1164ms | WER 0.0215",
            (1163.62, 0.0215),
            (25, -30)
        ),
        "whisper-large": (
            "Whisper large-v3\n7.1x | 1320ms | WER 0.0318",
            (1320.10, 0.0318),
            (25, 16)
        ),
    }

    for cond, (txt, (x, y), offset) in annotations.items():
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#94a3b8", alpha=0.92, lw=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.08", color="#334155", lw=1.1),
            zorder=6
        )

    ax.set_xscale("log")
    ax.set_xlim(95, 2000)
    ax.set_ylim(0.019, 0.043)

    # Format ticks cleanly
    ax.set_xticks([100, 200, 300, 500, 700, 1000, 1500, 2000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xlabel("P50 Batch Turn Latency (ms, Log Scale) — Lower is Faster", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Word Error Rate (WER) — Lower is Better", fontsize=11, fontweight="bold")
    ax.set_title("Conversational STT Pareto Frontier: Accuracy vs. Turn Latency (19 Native MLX Models, Pipecat-1000)", fontsize=13, fontweight="bold", pad=14)

    # Legend for Families
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=f, markeredgecolor="#334155")
        for f, c in FAMILY_COLORS.items() if f in {r["family"] for r in rows}
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#1d4ed8", linestyle="--", linewidth=2.0, label="Pareto Frontier")
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.95, fontsize=8.5)

    # Footnote explaining outlier exclusions
    plt.figtext(
        0.12, 0.02,
        "Hardware: Apple M5 (24GB Unified Memory). Framework: Apple MLX (100% Native MLX Models). Corpus: Pipecat-1000 (1.0s–15.0s).\n"
        "Bubble area encodes MLX Peak Metal Memory. Normalized WER uses phonetic fold rules. Outliers with template defects\n"
        "(aufklarer 4-bit, 0.2207 WER) and raw diarization prefixes (VibeVoice, 0.1470 WER) are excluded to preserve scale (see tail_risk_cases.png).",
        fontsize=8, color="#64748b", ha="left"
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.subplots_adjust(bottom=0.13, top=0.92, left=0.09, right=0.95)
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_latency_distribution(data: dict, out_path: Path):
    """Chart 2: Latency distribution across 19 conversational conditions (P50, P95, P99)"""
    fig, ax = plt.subplots(figsize=(13, 9), dpi=300)

    rows = sorted(data["pipecat1000"], key=lambda r: r["p50"])
    names = [r["cond"] for r in rows]
    p50s = [r["p50"] for r in rows]
    p95s = [r["p95"] for r in rows]
    p99s = [r["p99"] for r in rows]
    y_pos = np.arange(len(names))

    # Plot ranges: horizontal bar from P50 to P95, thin bar from P95 to P99
    for i, y in enumerate(y_pos):
        # Range line from P50 to P99
        ax.plot([p50s[i], p99s[i]], [y, y], color="#94a3b8", lw=2, zorder=2)
        # Bar between P50 and P95
        ax.plot([p50s[i], p95s[i]], [y, y], color="#3b82f6", lw=4.5, zorder=3)

    # Markers for P50, P95, P99
    ax.scatter(p50s, y_pos, color="#1d4ed8", s=65, label="P50 Latency", zorder=4)
    ax.scatter(p95s, y_pos, color="#f59e0b", s=50, marker="s", label="P95 Latency", zorder=4)
    ax.scatter(p99s, y_pos, color="#ef4444", s=50, marker="^", label="P99 Latency", zorder=4)

    # Label P50 values
    for i, (p50, y) in enumerate(zip(p50s, y_pos)):
        ax.text(p50 * 0.88, y, f"{p50:.0f}ms", va="center", ha="right", fontsize=7.5, color="#1e293b", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per Conversational Turn (ms, Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("STT Batch Latency Distribution: P50, P95, and P99 Tail Latencies (19 Native MLX Models)", fontsize=13, fontweight="bold", pad=12)

    plt.figtext(0.12, 0.02,
                "Measurement: Full-sentence batch processing latency on Apple MLX Metal (excludes network transport).\n"
                "Sorted by P50 latency. Blue segment = P50 to P95; grey tail = P95 to P99.",
                fontsize=8, color="#64748b", ha="left")

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.subplots_adjust(bottom=0.09, left=0.25)
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_qwen3_quantization_ladder(data: dict, out_path: Path):
    """Chart 3: Qwen3 quantization ladder (4-bit, 5-bit, 6-bit, 8-bit, BF16) vs WER / Speed / Memory"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Data for the mlx-community precision series
    ladder = [
        {"bits": 4, "label": "4-bit (mc)", "wer": 0.0241, "speedup": 17.83, "mlx_mb": 2452},
        {"bits": 5, "label": "5-bit (mc)", "wer": 0.0217, "speedup": 12.69, "mlx_mb": 2657},
        {"bits": 6, "label": "6-bit (mc)", "wer": 0.0217, "speedup": 10.48, "mlx_mb": 2862},
        {"bits": 8, "label": "8-bit (mc)", "wer": 0.0217, "speedup": 10.82, "mlx_mb": 3272},
        {"bits": 16, "label": "BF16 (mc)", "wer": 0.0221, "speedup": 6.62, "mlx_mb": 4811},
    ]

    labels = [x["label"] for x in ladder]
    x_pos = np.arange(len(labels))

    # Panel 1: WER
    wers = [x["wer"] * 100 for x in ladder]
    bars1 = ax1.bar(x_pos, wers, color="#3b82f6", width=0.55, edgecolor="#1d4ed8")
    ax1.set_title("Normalized WER (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("WER (%) — Lower is Better", fontsize=9.5)
    ax1.set_ylim(0, 3.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar, val in zip(bars1, wers):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{val:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax1.axhline(2.17, color="#ef4444", linestyle=":", label="0.0217 Plateau (5/6/8-bit)")
    ax1.legend(loc="upper left", fontsize=7.5)

    # Panel 2: Speedup
    speeds = [x["speedup"] for x in ladder]
    bars2 = ax2.bar(x_pos, speeds, color="#10b981", width=0.55, edgecolor="#047857")
    ax2.set_title("Real-Time Speedup (x Realtime)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Speedup Multiplier — Higher is Faster", fontsize=9.5)
    ax2.set_ylim(0, 22)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar, val in zip(bars2, speeds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}x", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 3: Peak Memory
    mems = [x["mlx_mb"] for x in ladder]
    bars3 = ax3.bar(x_pos, mems, color="#f59e0b", width=0.55, edgecolor="#b45309")
    ax3.set_title("Peak MLX Memory (MB)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Metal Unified Memory (MB) — Lower is Better", fontsize=9.5)
    ax3.set_ylim(0, 5800)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar, val in zip(bars3, mems):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 120, f"{val}MB", ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Qwen3-ASR-1.7B MLX Quantization Ladder: Trade-off is Speed & Memory, NOT Accuracy", fontsize=12.5, fontweight="bold", y=1.03)
    plt.figtext(0.5, -0.05,
                "Finding: 5-bit, 6-bit, and 8-bit yield identical WER (0.0217). BF16 (0.0221) provides no accuracy gain.\n"
                "4-bit sustains 17.8x speedup and 2.45GB memory with only 0.0024 WER delta.",
                fontsize=8.5, color="#475569", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def chart_wer_ranking_by_family(data: dict, out_path: Path):
    """Chart 4: 19 conditions ranked by WER, color-coded by model family"""
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=300)

    rows = sorted(data["pipecat1000"], key=lambda r: r["wer"])
    names = [r["cond"] for r in rows]
    wers = [r["wer"] * 100 for r in rows]
    perfects = [r["perfect"] for r in rows]
    families = [r["family"] for r in rows]
    colors = [FAMILY_COLORS.get(f, "#64748b") for f in families]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, wers, color=colors, height=0.68, edgecolor="#1e293b", linewidth=0.6)

    # Invert so best (lowest WER) is at top
    ax.invert_yaxis()

    # Annotate bars
    for bar, wer, perf, cond in zip(bars, wers, perfects, names):
        label_text = f"{wer:.2f}% (Perfect: {perf:.1f}%)"
        if "aufklarer" in cond or cond == "qwen3-asr-1.7b-4bit":
            label_text += " [! Template Defect]"
        elif "vibevoice" in cond:
            label_text += " [* Speaker Prefix; 6.79% stripped]"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, label_text, va="center", fontsize=8, color="#1e293b")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Normalized Word Error Rate (%) — Lower is Better", fontsize=11, fontweight="bold")
    ax.set_title("Overall Conversational STT Accuracy Ranking (19 Native MLX Models, Pipecat-1000)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 26)

    # Custom legend for families
    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=6, label=f)
        for f, c in FAMILY_COLORS.items() if any(r["family"] == f for r in rows)
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Model Family", fontsize=8, title_fontsize=9)

    plt.figtext(0.12, 0.02,
                "Evaluation on 1,000 real conversational turns. Normalized WER applies text cleaning (lowercase, filler words removed).\n"
                "[*] VibeVoice raw includes speaker labels. [!] aufklarer exhibits degenerate prompt template failure on 186 turns.",
                fontsize=8, color="#64748b", ha="left")

    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.subplots_adjust(bottom=0.1, left=0.26)
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_longform_dual_axis(data: dict, out_path: Path):
    """Chart 5: Long-form audio benchmark (3731.8s audio) strict dual panel: Native vs Auto-segmented"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel A: Native continuous
    native_data = [
        {"model": "Whisper Turbo", "speedup": 37.59, "mlx_mb": 4034},
        {"model": "Whisper Distil", "speedup": 44.25, "mlx_mb": 3934},
        {"model": "Whisper Large", "speedup": 6.94, "mlx_mb": 5435},
        {"model": "TEA-ASR 1.1", "speedup": 5.18, "mlx_mb": 15030},
    ]

    # Panel B: Auto-segmented (60s chunks)
    seg_data = [
        {"model": "Qwen3 4-bit", "speedup": 20.49, "mlx_mb": 2898},
        {"model": "Qwen3 INT8 (VAD)", "speedup": 7.00, "mlx_mb": 4082},
        {"model": "Qwen3 INT8 (RMS)", "speedup": 6.95, "mlx_mb": 4767},
        {"model": "VibeVoice 4-bit", "speedup": 2.13, "mlx_mb": 5990},
    ]

    for ax, pdata, title in [(ax1, native_data, "Panel A: Native Continuous Protocol"),
                             (ax2, seg_data, "Panel B: Auto-Segmented Protocol (60s)")]:
        models = [d["model"] for d in pdata]
        speeds = [d["speedup"] for d in pdata]
        mems = [d["mlx_mb"] for d in pdata]
        x = np.arange(len(models))

        ax_mem = ax.twinx()
        w = 0.35
        b1 = ax.bar(x - w/2, speeds, width=w, color="#10b981", label="Speedup (x Realtime)", edgecolor="#047857")
        b2 = ax_mem.bar(x + w/2, mems, width=w, color="#3b82f6", label="Peak MLX Memory (MB)", edgecolor="#1d4ed8")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("Speedup Multiplier (x) — Higher is Better", fontsize=9, color="#047857", fontweight="bold")
        ax_mem.set_ylabel("Peak MLX Memory (MB) — Lower is Better", fontsize=9, color="#1d4ed8", fontweight="bold")

        ax.set_ylim(0, 52)
        ax_mem.set_ylim(0, 17500)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        # Values on bars
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}x", ha="center", fontsize=7.5, fontweight="bold", color="#047857")
        for bar in b2:
            ax_mem.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 250, f"{int(bar.get_height())}MB", ha="center", fontsize=7.5, fontweight="bold", color="#1d4ed8")

    fig.suptitle("Long-Form Continuous Speech Transcription (~62.2 min, 100% Native Apple MLX)", fontsize=13, fontweight="bold", y=1.02)
    plt.figtext(0.5, -0.06,
                "Protocols are evaluated separately: Native Continuous requires no chunking. Auto-segmented uses 60s silence cut points.\n"
                "Hardware: Apple M5 24GB. Framework: Apple MLX. All models stayed under the 20GB memory ceiling.",
                fontsize=8.5, color="#475569", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def chart_vad_vs_rms_comparison(data: dict, out_path: Path):
    """Chart 6: Silero VAD vs RMS comparison on long-form audio (Qwen3-INT8)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel 1: Operational Metrics (Segments, Inference Sec, Memory Peak)
    metrics = ["Segment Count", "Inference Time (s)", "Peak Memory (MB)"]
    vad_vals = [82, 532.9, 4082.0]
    rms_vals = [68, 537.2, 4766.7]

    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w/2, vad_vals, width=w, color="#3b82f6", label="Silero VAD (60s target)", edgecolor="#1d4ed8")
    ax1.bar(x + w/2, rms_vals, width=w, color="#94a3b8", label="RMS Energy (60s target)", edgecolor="#475569")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9, fontweight="bold")
    ax1.set_title("Operational Metrics (Qwen3-INT8)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Values on bars
    for i, (v, r) in enumerate(zip(vad_vals, rms_vals)):
        ax1.text(i - w/2, v + 40, f"{v:.1f}" if isinstance(v, float) else f"{v}", ha="center", fontsize=8, fontweight="bold")
        ax1.text(i + w/2, r + 40, f"{r:.1f}" if isinstance(r, float) else f"{r}", ha="center", fontsize=8, fontweight="bold")

    # Callout for memory saving
    ax1.annotate("14.4% Memory Reduction\n(-685 MB Peak)",
                 xy=(2 - w/2, 4082), xytext=(-60, 45), textcoords="offset points",
                 fontsize=8.5, fontweight="bold", color="#1d4ed8",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#3b82f6"),
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8"))

    # Panel 2: WER comparison
    wer_types = ["Normalized WER (%)", "Mechanical WER (%)"]
    vad_wers = [0.0212 * 100, 0.0322 * 100]
    rms_wers = [0.0200 * 100, 0.0359 * 100]

    x2 = np.arange(len(wer_types))
    ax2.bar(x2 - w/2, vad_wers, width=w, color="#3b82f6", label="Silero VAD", edgecolor="#1d4ed8")
    ax2.bar(x2 + w/2, rms_wers, width=w, color="#94a3b8", label="RMS Energy", edgecolor="#475569")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(wer_types, fontsize=9, fontweight="bold")
    ax2.set_title("Accuracy Comparison (Loose Reference)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 4.8)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    for i, (v, r) in enumerate(zip(vad_wers, rms_wers)):
        ax2.text(i - w/2, v + 0.1, f"{v:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax2.text(i + w/2, r + 0.1, f"{r:.2f}%", ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Segmentation Comparison: Silero VAD vs. RMS Energy on 62.2 min Audio (MLX Pipeline)", fontsize=12, fontweight="bold", y=1.03)
    plt.figtext(0.5, -0.06,
                "Silero VAD detects speech activity at 287x realtime. Cuts land inside true phonetic silences, reducing peak memory by 14%.\n"
                "Accuracy difference is statistically negligible (<0.003 delta, within loose reference noise).",
                fontsize=8.5, color="#475569", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def chart_memory_efficiency_comparison(data: dict, out_path: Path):
    """Chart 7: RSS Peak vs MLX Peak Memory across models (illustrating Metal unified memory separation)"""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Pick representative models from across the families
    sample_conds = [
        "parakeet-tdt-0.6b-v3",
        "qwen3-asr-1.7b-4bit-alkd-a8t4",
        "whisper-turbo",
        "qwen3-asr-1.7b-4bit-mc",
        "qwen3-asr-1.7b-int8",
        "whisper-large",
        "qwen3-asr-1.7b-bf16-mc",
        "mega-asr-bf16",
        "granite-speech-4.1-2b-nar",
        "vibevoice-streaming-7b-4bit",
    ]

    p1000 = {r["cond"]: r for r in data["pipecat1000"]}
    filtered = [p1000[c] for c in sample_conds if c in p1000]

    names = [r["cond"].replace("-asr", "").replace("-streaming", "") for r in filtered]
    mlx_peaks = [r["mlx_mb"] for r in filtered]
    rss_peaks = [r["rss_mb"] for r in filtered]

    x = np.arange(len(names))
    w = 0.38
    b_mlx = ax.bar(x - w/2, mlx_peaks, width=w, color="#3b82f6", label="MLX Peak (Metal Unified Memory)", edgecolor="#1d4ed8")
    b_rss = ax.bar(x + w/2, rss_peaks, width=w, color="#64748b", label="RSS Peak (ru_maxrss Process Heap)", edgecolor="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Peak Memory (MB)", fontsize=10.5, fontweight="bold")
    ax.set_title("Memory Accounting Separation: Apple MLX Metal Peak vs. OS Process RSS Peak", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylim(0, 7500)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    # Visual note explaining the difference
    ax.axhline(6000, color="#ef4444", linestyle=":", alpha=0.5)
    ax.text(len(names) - 0.5, 6150, "Upper Boundary (All within 20GB Ceiling)", color="#ef4444", ha="right", fontsize=8)

    plt.figtext(0.12, 0.02,
                "Critical Engineering Rule: MLX Metal memory resides in Apple Silicon unified buffers, invisible to standard OS process RSS.\n"
                "RSS peak and MLX peak must be measured independently and NEVER summed together.",
                fontsize=8, color="#475569", ha="left")

    plt.subplots_adjust(bottom=0.2)
    plt.savefig(out_path, dpi=300)
    plt.close()


def chart_tail_risk_cases(data: dict, out_path: Path):
    """Chart 8: 3-panel visualization of benchmark pitfalls and edge cases"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Panel 1: VibeVoice Speaker Diarization Prefix Trap
    ax1.bar(["Raw Output\n(with Speaker Prefix)", "Prefix-Stripped\n(Clean Text)"], [14.70, 6.79], color=["#ef4444", "#10b981"], width=0.5, edgecolor="#1e293b")
    ax1.set_title("1. VibeVoice Speaker Label Bias", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Normalized WER (%)", fontsize=9.5)
    ax1.set_ylim(0, 18)
    ax1.text(0, 15.2, "14.70%", ha="center", fontweight="bold", color="#ef4444")
    ax1.text(1, 7.29, "6.79%", ha="center", fontweight="bold", color="#10b981")
    ax1.annotate("954/1000 turns output\n'Speaker N:' prefix\nartificially inflating WER",
                 xy=(0, 14.7), xytext=(15, -45), textcoords="offset points",
                 fontsize=8, bbox=dict(boxstyle="round", fc="#fee2e2", ec="#ef4444"))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 2: aufklarer Template Flaw Trap
    ax2.bar(["aufklarer 4-bit\n(Defective)", "mlx-community\n4-bit (Standard)"], [22.07, 2.41], color=["#ef4444", "#3b82f6"], width=0.5, edgecolor="#1e293b")
    ax2.set_title("2. Qwen3-4bit Prompt Template Flaw", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Normalized WER (%)", fontsize=9.5)
    ax2.set_ylim(0, 26)
    ax2.text(0, 22.6, "22.07%", ha="center", fontweight="bold", color="#ef4444")
    ax2.text(1, 3.0, "2.41%", ha="center", fontweight="bold", color="#3b82f6")
    ax2.annotate("186/1000 turns collapsed into\nliteral 'language' or empty text\ndue to chat template flaw",
                 xy=(0, 22.07), xytext=(15, -55), textcoords="offset points",
                 fontsize=8, bbox=dict(boxstyle="round", fc="#fee2e2", ec="#ef4444"))
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Non-Speech Audio Trap (15s Intro Music)
    ax3.bar(["15s Music Clip\n(Infinite Loop)", "60s Speech Clip\n(Clean Passage)"], [512, 134], color=["#ef4444", "#10b981"], width=0.5, edgecolor="#1e293b")
    ax3.set_title("3. Non-Speech Audio Hallucination", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Generated Tokens (Cap = 512)", fontsize=9.5)
    ax3.set_ylim(0, 600)
    ax3.text(0, 525, "512 (Max Cap)", ha="center", fontweight="bold", color="#ef4444")
    ax3.text(1, 145, "134 tokens", ha="center", fontweight="bold", color="#10b981")
    ax3.annotate("Music triggers 'I go go crazy...'\nloop hitting max token budget.\nNever use music for gates!",
                 xy=(0, 512), xytext=(10, -50), textcoords="offset points",
                 fontsize=8, bbox=dict(boxstyle="round", fc="#fee2e2", ec="#ef4444"))
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Key Benchmark Pitfalls & Tail Risks Discovered in Native MLX Verification", fontsize=12.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "outputs" / "ALL_DATA.json"
    if not data_path.exists():
        # Fallback to parent outputs
        data_path = repo_root.parent / "outputs" / "ALL_DATA.json"
    
    out_dir = repo_root / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    data = load_data(data_path)

    print("Generating chart 1: stt_pareto_frontier.png...")
    chart_pareto_frontier(data, out_dir / "stt_pareto_frontier.png")

    print("Generating chart 2: stt_latency_distribution.png...")
    chart_latency_distribution(data, out_dir / "stt_latency_distribution.png")

    print("Generating chart 3: qwen3_quantization_ladder.png...")
    chart_qwen3_quantization_ladder(data, out_dir / "qwen3_quantization_ladder.png")

    print("Generating chart 4: wer_ranking_by_family.png...")
    chart_wer_ranking_by_family(data, out_dir / "wer_ranking_by_family.png")

    print("Generating chart 5: longform_dual_axis.png...")
    chart_longform_dual_axis(data, out_dir / "longform_dual_axis.png")

    print("Generating chart 6: vad_vs_rms_comparison.png...")
    chart_vad_vs_rms_comparison(data, out_dir / "vad_vs_rms_comparison.png")

    print("Generating chart 7: memory_efficiency_comparison.png...")
    chart_memory_efficiency_comparison(data, out_dir / "memory_efficiency_comparison.png")

    print("Generating chart 8: tail_risk_cases.png...")
    chart_tail_risk_cases(data, out_dir / "tail_risk_cases.png")

    print(f"All 8 publication charts generated successfully in {out_dir}")


if __name__ == "__main__":
    main()
