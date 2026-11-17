#!/usr/bin/env python3
"""
plot_results.py
===============
Cache-Aware Parallel k-Means – Visualisation (Organised Output)
Author : Ganesh Yogeshwar Gadhave  |  Reg. No.: 241080018
Course : Parallel Computing [Batch A]

Output layout:
  plots/
  ├── baseline/           Plots that focus on baseline solver behaviour
  │   ├── b1_time_per_iteration.png
  │   ├── b2_convergence.png
  │   └── b3_cache_miss_baseline.png
  ├── improved/           Plots that focus on improved solver behaviour
  │   ├── i1_pruning_efficiency.png
  │   ├── i2_iteration_time_improved.png
  │   ├── i3_l1_tiling_effect.png
  │   └── i4_memory_footprint.png
  └── comparison/         Cross-solver / architecture-level comparisons
      ├── c1_summary_4panel.png
      ├── c2_speedup_waterfall.png
      ├── c3_scaling_curve.png
      ├── c4_all_convergence.png
      ├── c5_cache_vs_speedup.png
      ├── c6_pipeline_diagram.png
      └── c7_radar.png
"""

import csv, os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "legend.framealpha": 0.85,
})

# Colour palette – consistent across all plots
C_SEQ    = "#2c7bb6"   # 1 Sequential
C_NAIVE  = "#d7191c"   # 2 Naive OpenMP
C_PS     = "#fdae61"   # 3 Pruned Static
C_PG     = "#74add1"   # 4 Pruned Guided
C_PROP   = "#1a9641"   # 5 Proposed

SOLVER_COLORS = [C_SEQ, C_NAIVE, C_PS, C_PG, C_PROP]
SOLVER_SHORT  = ["Seq.\nBaseline", "Naive\nOpenMP", "+Pruning\n(Static)",
                 "+Guided\nSched.", "Proposed\n(Full)"]

BASELINE_SOLVERS = {
    "1. Sequential Baseline",
    "2. Naive OpenMP (Static + 8MB Buffers)",
}
IMPROVED_SOLVERS = {
    "3. + Low-Memory Geometric Pruning (Static)",
    "4. + Guided Dynamic Scheduling",
    "5. Proposed (+ 2KB L1 Tiling & SIMD)",
}

PLOT_DIR = {
    "baseline":    os.path.join("plots", "baseline"),
    "improved":    os.path.join("plots", "improved"),
    "comparison":  os.path.join("plots", "comparison"),
}
for d in PLOT_DIR.values():
    os.makedirs(d, exist_ok=True)

def save(fig, category, filename):
    path = os.path.join(PLOT_DIR[category], filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_summary(path="benchmark_summary.csv"):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k != "solver" else v)
                         for k, v in r.items()})
    return rows

def load_iterations(path="benchmark_iterations.csv"):
    data = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = r["solver"]
            if s not in data: data[s] = []
            data[s].append({
                "iteration":   int(r["iteration"]),
                "iter_time_s": float(r["iter_time_s"]),
                "inertia":     float(r["inertia"]),
                "pruned_pct":  float(r["pruned_pct"]),
            })
    return data

def load_scaling(path="benchmark_scaling.csv"):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"threads": int(r["threads"]),
                         "time_s": float(r["time_s"]),
                         "speedup": float(r["speedup"])})
    return rows

# ===========================================================================
#  BASELINE PLOTS
# ===========================================================================

def b1_time_per_iteration(iter_data):
    """Baseline solvers: per-iteration wall-clock time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for solver, color in [(k, SOLVER_COLORS[i])
                          for i, k in enumerate(list(iter_data.keys()))
                          if k in BASELINE_SOLVERS]:
        rows  = iter_data[solver]
        iters = [r["iteration"]   for r in rows]
        times = [r["iter_time_s"] for r in rows]
        ax.plot(iters, times, "o-", color=color, label=solver,
                linewidth=2.2, markersize=7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Iteration Wall-Clock Time (s)")
    ax.set_title("Baseline Solvers – Per-Iteration Execution Time\n"
                 "Sequential vs Naive OpenMP (static schedule + critical section)",
                 fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "baseline", "b1_time_per_iteration.png")

def b2_convergence_baseline(iter_data):
    """Baseline solvers: inertia convergence."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (solver, color) in enumerate(
            [(k, SOLVER_COLORS[i]) for i, k in enumerate(iter_data.keys())
             if k in BASELINE_SOLVERS]):
        rows    = iter_data[solver]
        iters   = [r["iteration"] for r in rows]
        inertia = [r["inertia"]   for r in rows]
        ax.plot(iters, inertia, "o-", color=color, label=solver,
                linewidth=2.2, markersize=7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Inertia (Sum of Squared Distances)")
    ax.set_title("Baseline Solvers – Convergence\n"
                 "Both reach same quality; difference is wall-clock speed",
                 fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "baseline", "b2_convergence.png")

def b3_cache_miss_baseline(summary):
    """Baseline: cache miss rate bar chart showing thrashing problem."""
    base_rows = [r for r in summary if r["solver"] in BASELINE_SOLVERS]
    names  = [r["solver"]         for r in base_rows]
    cache  = [r["cache_miss_rate"] for r in base_rows]
    colors = [C_SEQ, C_NAIVE]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(names)), cache, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    for bar, c in zip(bars, cache):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{c:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Sequential\nBaseline", "Naive OpenMP\n(Static + Critical)"],
                       fontsize=10)
    ax.set_ylabel("L2/L3 Cache Miss Rate (%)")
    ax.set_title("⚠ Cache Thrashing Problem – Baseline Solvers\n"
                 "Naive parallelism with 8MB thread-private buffers triples cache misses",
                 fontweight="bold")
    ax.set_ylim(0, 50)

    # Annotation arrow showing problem
    ax.annotate("Cache thrashing!\n(8 MB private buffers\noverflow L2/L3)",
                xy=(1, cache[1]), xytext=(1.3, cache[1] + 5),
                fontsize=8.5, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    plt.tight_layout()
    save(fig, "baseline", "b3_cache_miss_baseline.png")

# ===========================================================================
#  IMPROVED PLOTS
# ===========================================================================

def i1_pruning_efficiency(iter_data):
    """Improved: pruning % per iteration for all pruning solvers."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pruning_map = {
        "3. + Low-Memory Geometric Pruning (Static)": C_PS,
        "4. + Guided Dynamic Scheduling":             C_PG,
        "5. Proposed (+ 2KB L1 Tiling & SIMD)":      C_PROP,
    }
    for solver, color in pruning_map.items():
        if solver not in iter_data: continue
        rows   = iter_data[solver]
        iters  = [r["iteration"]  for r in rows]
        pruned = [r["pruned_pct"] for r in rows]
        ax.plot(iters, pruned, "o-", color=color, label=solver,
                linewidth=2.2, markersize=7)

    ax.axhline(78.0, color="gray", linestyle=":", linewidth=1.5,
               label="78% avg (paper target)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance Evaluations Pruned (%)")
    ax.set_title("Geometric Pruning Efficiency – Improved Solvers\n"
                 "Two-level triangle-inequality filter: ~85% evaluations skipped",
                 fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    save(fig, "improved", "i1_pruning_efficiency.png")

def i2_iteration_time_improved(iter_data):
    """Improved: per-iteration time showing speed gains."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    improved_map = {
        "3. + Low-Memory Geometric Pruning (Static)": C_PS,
        "4. + Guided Dynamic Scheduling":             C_PG,
        "5. Proposed (+ 2KB L1 Tiling & SIMD)":      C_PROP,
    }
    for solver, color in improved_map.items():
        if solver not in iter_data: continue
        rows  = iter_data[solver]
        iters = [r["iteration"]   for r in rows]
        times = [r["iter_time_s"] for r in rows]
        ax.plot(iters, times, "o-", color=color, label=solver,
                linewidth=2.2, markersize=7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Iteration Wall-Clock Time (s)")
    ax.set_title("Improved Solvers – Per-Iteration Time\n"
                 "Pruning warms up: later iterations are progressively faster",
                 fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    save(fig, "improved", "i2_iteration_time_improved.png")

def i3_l1_tiling_effect(summary):
    """Improved: bar chart isolating the L1-tiling cache-miss improvement."""
    # Focus on solvers 4 and 5 (same pruning, only L1-tiling differs)
    targets = {
        "4. + Guided Dynamic Scheduling":        ("Without L1-Tiling", C_PG),
        "5. Proposed (+ 2KB L1 Tiling & SIMD)": ("With L1-Tiling",    C_PROP),
    }
    rows = [r for r in summary if r["solver"] in targets]
    if len(rows) < 2:
        return

    labels = [targets[r["solver"]][0] for r in rows]
    colors = [targets[r["solver"]][1] for r in rows]
    cache  = [r["cache_miss_rate"]    for r in rows]
    times  = [r["time_s"]             for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Cache miss
    b1 = ax1.bar(range(2), cache, color=colors, edgecolor="black",
                 linewidth=0.8, width=0.4)
    for bar, c in zip(b1, cache):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{c:.1f}%", ha="center", fontsize=11, fontweight="bold")
    reduction_pct = (cache[0] - cache[1]) / cache[0] * 100
    ax1.set_xticks(range(2)); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("L2/L3 Cache Miss Rate (%)")
    ax1.set_title(f"L1-Tiling Effect on Cache Misses\n"
                  f"↓ {reduction_pct:.0f}% reduction",
                  fontweight="bold")
    ax1.set_ylim(0, max(cache) * 1.35)

    # Time
    b2 = ax2.bar(range(2), times, color=colors, edgecolor="black",
                 linewidth=0.8, width=0.4)
    for bar, t in zip(b2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{t:.2f}s", ha="center", fontsize=11, fontweight="bold")
    gain = (times[0] - times[1]) / times[0] * 100
    ax2.set_xticks(range(2)); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Total Execution Time (s)")
    ax2.set_title(f"L1-Tiling Effect on Execution Time\n"
                  f"↓ {gain:.0f}% faster",
                  fontweight="bold")
    ax2.set_ylim(0, max(times) * 1.35)

    plt.tight_layout()
    save(fig, "improved", "i3_l1_tiling_effect.png")

def i4_memory_footprint():
    """Improved: memory footprint comparison (log scale)."""
    configs = [
        ("Naive Thread\nBuffers\n(T×K×D)",      8.38,  C_NAIVE),
        ("Yinyang\nLower Bounds\n(N×K)",         102.4, "#fdae61"),
        ("Proposed\nUpper Bounds u[N]",          0.38,  C_PG),
        ("Proposed\nPairwise S[K×K]",            0.25,  C_PS),
        ("Proposed\nL1 Accumulator\n(2KB/thread)", 0.002, C_PROP),
    ]
    labels = [c[0] for c in configs]
    sizes  = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(labels)), sizes, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.6)
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2,
                sz * 1.5, f"{sz:.3f} MB",
                ha="center", fontsize=8.5, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Memory Size (MB, log scale)")
    ax.set_title("Improved Solver – Memory Footprint of Key Data Structures\n"
                 "O(N) upper bounds replace O(N×K) lower bounds: 102 MB → 0.38 MB",
                 fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)

    ax.axhline(4.0,  color="#888", linestyle="--", linewidth=1.2,
               label="Typical L2 Cache (4 MB)")
    ax.axhline(32.0, color="brown", linestyle=":", linewidth=1.2,
               label="Typical L3 Cache (32 MB)")
    ax.legend(fontsize=8.5)

    # Bracket showing proposed footprint
    ax.annotate("", xy=(2, 0.38), xytext=(4, 0.38),
                arrowprops=dict(arrowstyle="<->", color=C_PROP, lw=2))
    ax.text(3, 0.5, "All fit in L2/L1!", ha="center", fontsize=8.5,
            color=C_PROP, fontweight="bold")

    plt.tight_layout()
    save(fig, "improved", "i4_memory_footprint.png")

# ===========================================================================
#  COMPARISON PLOTS
# ===========================================================================

def c1_summary_4panel(summary):
    """4-panel paper Fig-1 style comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Cache-Aware Parallel k-Means – Full Benchmark Summary\n"
        "N=100,000  D=512  K=256  Iterations=15  (24-core CPU)",
        fontsize=13, fontweight="bold", y=1.01
    )
    names  = [r["solver"]           for r in summary]
    times  = [r["time_s"]           for r in summary]
    speeds = [r["speedup"]          for r in summary]
    evals  = [r["active_evals_pct"] for r in summary]
    cache  = [r["cache_miss_rate"]  for r in summary]
    xs     = np.arange(len(names))
    colors = SOLVER_COLORS[:len(names)]

    for ax, vals, title, ylabel, annot_fn in [
        (axes[0,0], times,  "(A) Execution Time (s)  — Lower is Better",
         "Wall-clock Time (s)", lambda v: f"{v:.1f}s"),
        (axes[0,1], speeds, "(B) Speedup Factor  — Higher is Better",
         "Speedup (×)", lambda v: f"{v:.1f}×"),
        (axes[1,0], evals,  "(C) Active Distance Evaluations %  — Lower is Better",
         "Active Eval %", lambda v: f"{v:.1f}%"),
        (axes[1,1], cache,  "(D) L2/L3 Cache Miss Rate %  — Lower is Better",
         "Cache Miss Rate (%)", lambda v: f"{v:.1f}%"),
    ]:
        bars = ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs); ax.set_xticklabels(SOLVER_SHORT[:len(names)], fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    annot_fn(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
        # Highlight proposed
        bars[-1].set_edgecolor("gold"); bars[-1].set_linewidth(2.5)

    plt.tight_layout()
    save(fig, "comparison", "c1_summary_4panel.png")

def c2_speedup_waterfall(summary):
    """Cumulative speedup waterfall."""
    names  = [r["solver"]  for r in summary]
    speeds = [r["speedup"] for r in summary]
    xs     = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(xs, speeds, color=SOLVER_COLORS[:len(names)],
                  edgecolor="black", linewidth=0.8, width=0.6)

    for i in range(1, len(speeds)):
        ratio = speeds[i] / speeds[i - 1]
        y = max(speeds[i-1], speeds[i]) + 0.5
        ax.annotate("", xy=(xs[i], y), xytext=(xs[i-1], y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.3))
        ax.text((xs[i-1]+xs[i])/2, y+0.2, f"+{ratio:.1f}×",
                ha="center", fontsize=8.5, color="gray")

    for bar, sp in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{sp:.1f}×", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel("Speedup over Sequential Baseline (×)")
    ax.set_title("Cumulative Speedup Waterfall – Each Bar Adds One Optimisation",
                 fontweight="bold")
    ax.set_ylim(0, max(speeds) * 1.3)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    plt.tight_layout()
    save(fig, "comparison", "c2_speedup_waterfall.png")

def c3_scaling_curve(scaling):
    """Thread-count scaling vs ideal."""
    threads  = [r["threads"] for r in scaling]
    speedups = [r["speedup"] for r in scaling]
    times    = [r["time_s"]  for r in scaling]
    ideal    = [float(t) / threads[0] for t in threads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(threads, ideal, "k--", label="Ideal Linear Speedup",
             linewidth=1.5, alpha=0.5)
    ax1.plot(threads, speedups, "o-", color=C_PROP, label="Proposed Solver",
             linewidth=2.3, markersize=8)
    ax1.fill_between(threads, speedups, ideal, alpha=0.1, color="gray",
                     label="Efficiency gap")
    ax1.set_xlabel("Number of CPU Threads")
    ax1.set_ylabel("Speedup (×) over 1-thread Proposed")
    ax1.set_title("Multi-Core Scaling Curve\nProposed Solver", fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.set_xlim(0, max(threads) + 1)
    ax1.set_ylim(0, max(max(ideal), max(speedups)) * 1.15)

    ax2.plot(threads, times, "s-", color=C_SEQ, linewidth=2.3, markersize=8)
    for t, tm in zip(threads, times):
        ax2.text(t, tm + 0.03, f"{tm:.2f}s", ha="center", fontsize=7.5)
    ax2.set_xlabel("Number of CPU Threads")
    ax2.set_ylabel("Execution Time (s)")
    ax2.set_title("Execution Time vs Thread Count", fontweight="bold")

    plt.tight_layout()
    save(fig, "comparison", "c3_scaling_curve.png")

def c4_all_convergence(iter_data):
    """All-solver inertia convergence."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, (solver, rows) in enumerate(iter_data.items()):
        iters   = [r["iteration"] for r in rows]
        inertia = [r["inertia"]   for r in rows]
        ax.plot(iters, inertia, "o-",
                color=SOLVER_COLORS[idx % len(SOLVER_COLORS)],
                label=solver, linewidth=2.0, markersize=5, alpha=0.9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Inertia (Sum of Squared Distances)")
    ax.set_title("All Solvers – Convergence Curve\n"
                 "Same quality, dramatically different speeds",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    save(fig, "comparison", "c4_all_convergence.png")

def c5_cache_vs_speedup(summary):
    """Cache miss vs speedup scatter."""
    cache  = [r["cache_miss_rate"] for r in summary]
    speeds = [r["speedup"]         for r in summary]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (c, s) in enumerate(zip(cache, speeds)):
        ax.scatter(c, s, s=180, color=SOLVER_COLORS[i],
                   edgecolors="black", linewidth=0.8, zorder=5)
        ax.annotate(SOLVER_SHORT[i],
                    xy=(c, s), xytext=(c + 0.5, s + 0.4),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    z = np.polyfit(cache, speeds, 1)
    xs = np.linspace(min(cache)-1, max(cache)+1, 100)
    ax.plot(xs, np.poly1d(z)(xs), "k--", linewidth=1.2, alpha=0.4)
    ax.set_xlabel("L2/L3 Cache Miss Rate (%)")
    ax.set_ylabel("Speedup (×) over Sequential")
    ax.set_title("Cache Miss Rate vs Speedup\nLower cache misses → higher speedup",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, "comparison", "c5_cache_vs_speedup.png")

def c6_pipeline_diagram():
    """Annotated four-stage pipeline block diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")

    stages = [
        ("Stage 1\nAligned Init",
         "• 64-byte aligned alloc\n• Flat 1D buffers X[N×D]\n• Precompute ‖xᵢ‖²",
         "#4575b4", 0.5),
        ("Stage 2\nDrift & Bounds",
         "• drift[k]=‖Cₖ(t)−Cₖ(t-1)‖\n• S[a][b]=½d(Cₐ,C_b)\n• O(N) upper bounds",
         "#74add1", 3.5),
        ("Stage 3\nPruned Assignment",
         "• schedule(guided,64)\n• Level 1: Ball filter\n• Level 2: △-ineq\n• Lock-free writes",
         "#a6d96a", 6.5),
        ("Stage 4\nL1-Resident Reduction",
         "• CSR cluster index\n• local_sum[512]=2KB\n• Fits in 32KB L1\n• Zero atomic locks",
         "#1a9641", 9.5),
    ]
    for title, detail, color, x0 in stages:
        rect = mpatches.FancyBboxPatch(
            (x0, 1.5), 2.6, 3.0, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.88)
        ax.add_patch(rect)
        ax.text(x0+1.3, 4.1, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x0+1.3, 2.8, detail, ha="center", va="center",
                fontsize=7.8, color="white", linespacing=1.5)
    for x in [3.1, 6.1, 9.1]:
        ax.annotate("", xy=(x+0.4, 3.0), xytext=(x, 3.0),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=2.0))
    ax.text(7, 5.65, "Four-Stage Cache-Aware Execution Pipeline",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(7, 0.6,
            "OpenMP Parallel  |  AVX2/AVX-512 SIMD  |  64-byte Aligned Memory  |  L1-Resident Accumulation",
            ha="center", fontsize=9, style="italic", color="#444")
    mem_labels = ["N×D float32\n400 KB (D=512)", "S[K×K]=256 KB\nu[N]=400 KB",
                  "Shared reads\nLock-free writes", "2 KB stack\nper thread"]
    for (_, _, _, x0), label in zip(stages, mem_labels):
        ax.text(x0+1.3, 1.1, label, ha="center", va="center", fontsize=7.5,
                color="#444",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#aaa", alpha=0.8))
    plt.tight_layout()
    save(fig, "comparison", "c6_pipeline_diagram.png")

def c7_radar(summary):
    """Multi-metric radar chart."""
    solvers = [r["solver"] for r in summary]
    max_sp  = max(r["speedup"] for r in summary)
    max_t   = max(r["time_s"]  for r in summary)
    metrics = []
    for r in summary:
        metrics.append([
            r["speedup"] / max_sp * 100,
            (1 - r["cache_miss_rate"] / 40) * 100,
            100 - r["active_evals_pct"],
            (1 - r["time_s"] / max_t) * 100,
        ])

    mnames = ["Speedup\n(norm.)", "Cache\nEfficiency",
              "Pruning\nEfficiency", "Time\nEfficiency"]
    N_m = len(mnames)
    angles = [n / N_m * 2 * math.pi for n in range(N_m)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2); ax.set_theta_direction(-1)

    for idx, (vals, solver) in enumerate(zip(metrics, solvers)):
        vals_p = vals + vals[:1]
        ax.plot(angles, vals_p, "o-", color=SOLVER_COLORS[idx],
                linewidth=2.0, markersize=5, label=solver, alpha=0.9)
        ax.fill(angles, vals_p, color=SOLVER_COLORS[idx], alpha=0.07)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(mnames, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Multi-Metric Radar – All 5 Solver Configurations\n(Outward = Better)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7.5)
    plt.tight_layout()
    save(fig, "comparison", "c7_radar.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("  Cache-Aware Parallel k-Means – Plot Generator")
    print("  Output: plots/baseline/  plots/improved/  plots/comparison/")
    print("=" * 62)

    for f in ["benchmark_summary.csv", "benchmark_iterations.csv",
              "benchmark_scaling.csv"]:
        if not os.path.exists(f):
            print(f"[ERROR] {f} not found. Run ./kmeans_bench first.")
            sys.exit(1)

    summary   = load_summary()
    iter_data = load_iterations()
    scaling   = load_scaling()

    print(f"\nLoaded: {len(summary)} solvers | {len(scaling)} scaling points\n")

    print("── BASELINE PLOTS ──────────────────────────────────────")
    b1_time_per_iteration(iter_data)
    b2_convergence_baseline(iter_data)
    b3_cache_miss_baseline(summary)

    print("\n── IMPROVED PLOTS ──────────────────────────────────────")
    i1_pruning_efficiency(iter_data)
    i2_iteration_time_improved(iter_data)
    i3_l1_tiling_effect(summary)
    i4_memory_footprint()

    print("\n── COMPARISON PLOTS ─────────────────────────────────────")
    c1_summary_4panel(summary)
    c2_speedup_waterfall(summary)
    c3_scaling_curve(scaling)
    c4_all_convergence(iter_data)
    c5_cache_vs_speedup(summary)
    c6_pipeline_diagram()
    c7_radar(summary)

    print(f"\n✓ All 14 plots saved.\n")
    for cat, d in PLOT_DIR.items():
        files = sorted(os.listdir(d))
        print(f"  plots/{cat}/  ({len(files)} files)")
        for fn in files:
            print(f"    {fn}")

if __name__ == "__main__":
    main()
