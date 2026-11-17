"""
make_docx.py
============
Converts the research paper into a properly formatted Word (.docx) document.
Includes: title page, abstract, all sections, tables, equations (as styled text),
and all 14 embedded plots.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTS = {
    "b1": r"c:\Sem 5 Labs\PC\Mini Project\plots\baseline\b1_time_per_iteration.png",
    "b2": r"c:\Sem 5 Labs\PC\Mini Project\plots\baseline\b2_convergence.png",
    "b3": r"c:\Sem 5 Labs\PC\Mini Project\plots\baseline\b3_cache_miss_baseline.png",
    "i1": r"c:\Sem 5 Labs\PC\Mini Project\plots\improved\i1_pruning_efficiency.png",
    "i2": r"c:\Sem 5 Labs\PC\Mini Project\plots\improved\i2_iteration_time_improved.png",
    "i3": r"c:\Sem 5 Labs\PC\Mini Project\plots\improved\i3_l1_tiling_effect.png",
    "i4": r"c:\Sem 5 Labs\PC\Mini Project\plots\improved\i4_memory_footprint.png",
    "c1": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c1_summary_4panel.png",
    "c2": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c2_speedup_waterfall.png",
    "c3": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c3_scaling_curve.png",
    "c4": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c4_all_convergence.png",
    "c5": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c5_cache_vs_speedup.png",
    "c6": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c6_pipeline_diagram.png",
    "c7": r"c:\Sem 5 Labs\PC\Mini Project\plots\comparison\c7_radar.png",
}
OUT = r"c:\Sem 5 Labs\PC\Mini Project\Research_Paper_Ganesh_241080018.docx"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"),   kwargs.get("val",   "single"))
        border.set(qn("w:sz"),    kwargs.get("sz",    "4"))
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), kwargs.get("color", "2C3E50"))
        tcBorders.append(border)
    tcPr.append(tcBorders)

def add_horizontal_rule(doc):
    p  = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pb  = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "2C3E50")
    pb.append(bot)
    pPr.append(pb)
    p.paragraph_format.space_after = Pt(6)

def add_figure(doc, img_key, caption, width=6.2):
    if img_key not in PLOTS or not os.path.exists(PLOTS[img_key]):
        p = doc.add_paragraph(f"[Figure: {caption}  — image not found]")
        p.runs[0].italic = True
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(PLOTS[img_key], width=Inches(width))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after  = Pt(10)
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

def styled_heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x52, 0x76)
    return p

def add_body(doc, text, space_after=6):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.space_before = Pt(2)
    p.style.font.size = Pt(11)
    return p

def add_equation(doc, equation_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(8)
    r = p.add_run(equation_text)
    r.font.name  = "Cambria Math"
    r.font.size  = Pt(11)
    r.font.bold  = False
    r.font.italic = True
    return p

def add_code_block(doc, code):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent   = Cm(1.0)
    p.paragraph_format.space_before  = Pt(4)
    p.paragraph_format.space_after   = Pt(8)
    p.paragraph_format.line_spacing  = Pt(14)
    r = p.add_run(code)
    r.font.name = "Courier New"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x1A, 0x5C, 0x2A)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"),   "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"),  "F0F4F0")
    p._p.get_or_add_pPr().append(shading)

def make_table(doc, headers, rows, col_widths=None, header_color="1A5276"):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    # Header row
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        set_cell_bg(hdr[i], header_color)
        hdr[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(10)
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.add_row().cells
        bg  = "EBF5FB" if ri % 2 == 0 else "FDFEFE"
        for i, cell_text in enumerate(row_data):
            row[i].text = str(cell_text)
            set_cell_bg(row[i], bg)
            row[i].paragraphs[0].runs[0].font.size = Pt(10)
            row[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for cell in table.columns[i].cells:
                cell.width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    return table

def add_callout(doc, text, style="note"):
    colors = {"note": "D6EAF8", "warn": "FDEBD0", "key": "D5F5E3"}
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.8)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(8)
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.italic = True
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"),   "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"),  colors.get(style, "D6EAF8"))
    p._p.get_or_add_pPr().append(shading)

# ===========================================================================
# BUILD DOCUMENT
# ===========================================================================
doc = Document()

# --- Page margins ---
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.8)
    section.right_margin  = Cm(2.8)

# ---------------------------------------------------------------------------
# TITLE PAGE
# ---------------------------------------------------------------------------
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Cache-Aware Parallel k-Means\nfor High-Dimensional AI Embeddings")
r.bold      = True
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(0x1A, 0x52, 0x76)
title.paragraph_format.space_after = Pt(16)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Parallel Computing Mini Project  |  Batch A")
r.font.size = Pt(12)
r.italic    = True
r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
sub.paragraph_format.space_after = Pt(10)

auth = doc.add_paragraph()
auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
for line in [
    ("Ganesh Yogeshwar Gadhave", True,  13),
    ("\nReg. No.: 241080018  |  Information Technology", False, 11),
    ("\nParallel Computing [Batch A]", False, 11),
]:
    r = auth.add_run(line[0])
    r.bold      = line[1]
    r.font.size = Pt(line[2])
auth.paragraph_format.space_after = Pt(20)

add_horizontal_rule(doc)

# ---------------------------------------------------------------------------
# ABSTRACT
# ---------------------------------------------------------------------------
styled_heading(doc, "Abstract", 1)
add_body(doc,
    "Vector quantization and k-means clustering serve as foundational indexing primitives "
    "in modern Retrieval-Augmented Generation (RAG) systems and vector databases (such as "
    "Meta FAISS and Milvus) to accelerate similarity search across millions of high-dimensional "
    "embeddings (D ≥ 512). While recent algorithmic advancements successfully reduce "
    "point-to-centroid distance evaluations, scaling these pruned algorithms across multi-core "
    "CPUs via OpenMP exposes severe parallel bottlenecks. Specifically, thread-private centroid "
    "accumulator buffers exceed L2/L3 cache capacities, causing cache thrashing during the "
    "centroid update phase, while static workload partitioning leads to severe thread load "
    "imbalance due to irregular distance pruning. This paper proposes a hardware-aware, "
    "cache-tiled, and lock-free parallel k-Means architecture that couples lightweight "
    "movement-bound filtering with guided dynamic scheduling and two-stage L1-resident cluster "
    "accumulation. Implemented in compact C++17 with OpenMP, the proposed solver achieves a "
    "26.86× speedup over the sequential baseline on 100,000 high-dimensional vectors "
    "(D = 512, K = 256) running on a 24-core commodity CPU — while simultaneously reducing "
    "the L2/L3 cache miss rate from 38.4% to 3.2% and skipping ~85% of all distance evaluations."
)

add_horizontal_rule(doc)
doc.add_page_break()

# ---------------------------------------------------------------------------
# SECTION 1 — INTRODUCTION
# ---------------------------------------------------------------------------
styled_heading(doc, "1. Introduction", 1)
add_body(doc,
    "The explosive growth of large language models (LLMs) has made vector databases a critical "
    "infrastructure component. Systems such as FAISS and Milvus partition billions of "
    "high-dimensional embedding vectors into coarse cluster centroids using k-means — a step "
    "called Inverted File (IVF) quantization. For modern embeddings (D ≥ 512), constructing "
    "these IVF indexes requires executing iterative k-means refinement over datasets with "
    "N ≥ 10⁶ vectors, making computational efficiency mission-critical."
)
add_body(doc,
    "The standard Lloyd's algorithm alternates between two phases per iteration: (1) Assignment — "
    "for each of the N vectors, find its nearest centroid among K candidates by evaluating "
    "N × K squared Euclidean distances; and (2) Update — recompute each centroid as the mean "
    "of its assigned vectors. For N = 10⁶, K = 500, D = 768, the assignment phase alone "
    "requires ~7.68 × 10¹¹ floating-point multiply-add operations per iteration."
)
add_body(doc,
    "Parallelizing this workload across modern multi-core CPUs using OpenMP is natural, "
    "but introduces three hardware-level bottlenecks that this paper resolves:"
)
add_callout(doc,
    "B1 — L2/L3 Cache Memory Wall: Thread-private centroid accumulator matrices of size "
    "T × K × D (e.g. 16 × 256 × 512 × 4 bytes ≈ 8.38 MB) overflow per-core L2 caches "
    "and saturate L3 memory bandwidth during the centroid update phase.", "warn")
add_callout(doc,
    "B2 — Dynamic Thread Load Imbalance: Distance pruning algorithms skip computations "
    "for stable interior vectors. Under static OpenMP scheduling, threads finishing early "
    "sit idle while boundary threads evaluate unpruned vectors.", "warn")
add_callout(doc,
    "B3 — Global State Contention: Synchronizing shared centroid accumulation via atomic "
    "or critical directives serializes thread memory writes, reducing parallel scaling to "
    "nearly single-core throughput during the update phase.", "warn")

styled_heading(doc, "1.1 Contributions", 2)
for contrib in [
    "A two-level triangle-inequality geometric filter that replaces O(N × K) lower-bound "
    "tables with an O(N) scalar upper-bound array.",
    "OpenMP guided dynamic scheduling [schedule(guided, 64)] to eliminate load imbalance "
    "from irregular pruning workloads.",
    "A two-stage L1-resident centroid reduction using per-cluster parallelism and 2 KB "
    "stack-local accumulators that fit entirely within each core's 32 KB L1 data cache "
    "with zero atomic locks.",
    "A complete C++17/OpenMP implementation with empirical benchmarks across five "
    "progressive solver configurations.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(contrib).font.size = Pt(11)

doc.add_paragraph()

# ---------------------------------------------------------------------------
# SECTION 2 — BACKGROUND
# ---------------------------------------------------------------------------
styled_heading(doc, "2. Background and Related Work", 1)

styled_heading(doc, "2.1 Standard k-Means and Brute-Force Complexity", 2)
add_body(doc,
    "Lloyd's algorithm iterates until convergence. The dominant cost is the assignment step: "
    "for each vector xᵢ ∈ ℝᴰ, the nearest centroid is found by evaluating all K distances. "
    "Naively, this requires O(N · K · D) multiply-add operations per iteration. For N = 100,000, "
    "K = 256, D = 512, each iteration processes approximately 1.31 × 10¹⁰ floating-point operations."
)

styled_heading(doc, "2.2 Geometric Distance Pruning", 2)
add_body(doc,
    "The triangle inequality has been exploited extensively for sub-quadratic nearest-neighbour "
    "search. The Yinyang algorithm extends single-bound filtering to group-level bounds but "
    "requires O(N × K) storage (102 MB for the above dimensions). Ball k-Means proposes "
    "center-based bounds without explicit lower-bound maintenance. The Angle k-Means method "
    "introduces angular distance constraints applicable to normalized embeddings. The key insight "
    "used in this work: maintaining one scalar upper bound u[i] ≈ d(xᵢ, C_assigned) per vector, "
    "updated via centroid drift δₖ = ‖Cₖ⁽ᵗ⁾ - Cₖ⁽ᵗ⁻¹⁾‖, enables a compact O(N) bound structure."
)

styled_heading(doc, "2.3 Parallel k-Means Approaches", 2)
make_table(doc,
    ["Approach", "Problem"],
    [
        ["#pragma omp atomic per dimension", "D atomic ops per vector — extreme overhead"],
        ["#pragma omp critical per vector",  "Full serialization — no parallelism"],
        ["Thread-private T×K×D buffers",     "Cache thrashing — 8.38 MB per 16 threads"],
        ["Lock-free write to separate arrays","Requires restructuring the update loop (our solution)"],
    ],
    col_widths=[3.0, 3.6]
)

# ---------------------------------------------------------------------------
# SECTION 3 — PROBLEM FORMULATION
# ---------------------------------------------------------------------------
styled_heading(doc, "3. Problem Formulation", 1)

styled_heading(doc, "3.1 Formal Objective", 2)
add_body(doc,
    "Given a dataset X = {x₁, x₂, …, xₙ} ⊂ ℝᴰ and cluster count K, k-means minimizes "
    "the total inertia (sum of squared distances to assigned centroids):"
)
add_equation(doc, "L(X, C) = Σᵢ min_{k ∈ [K]}  ‖xᵢ − Cₖ‖²")

styled_heading(doc, "3.2 Two-Level Triangle Inequality Filter", 2)
add_body(doc,
    "For vector xᵢ currently assigned to cluster C_A, we maintain a scalar upper bound "
    "u[i] ≈ d(xᵢ, C_A). At each iteration, the centroid drift updates the bound:"
)
add_equation(doc, "u[i] ← u[i] + drift[A],     drift[k] = ‖Cₖ⁽ᵗ⁾ − Cₖ⁽ᵗ⁻¹⁾‖")
add_body(doc, "We also maintain the pairwise centroid half-distance matrix:")
add_equation(doc, "S[a][b] = ½ · d(Cₐ, C_b)")
add_body(doc, "This enables two cascade filters before any O(D) vector operation:")
add_callout(doc,
    "Level 1 — Cluster Ball Filter:\n"
    "u[i] ≤ min_{B≠A}(S[A][B])  →  skip ALL K centroids for vector xᵢ", "key")
add_callout(doc,
    "Level 2 — Pairwise Triangle Inequality:\n"
    "d(xᵢ, C_A) < S[A][B] − δ_max  →  skip centroid C_B", "key")

styled_heading(doc, "3.3 SIMD Dot-Product Distance Expansion", 2)
add_body(doc,
    "When a candidate centroid survives both filters, the squared Euclidean distance is "
    "computed via algebraic expansion to enable SIMD vectorization:"
)
add_equation(doc, "‖xᵢ − C_B‖² = ‖xᵢ‖² + ‖C_B‖² − 2 Σ_{d=0}^{D−1} xᵢ,d · C_B,d")
add_body(doc,
    "Precomputing ‖xᵢ‖² once per vector and ‖Cₖ‖² per centroid per iteration, the inner "
    "loop reduces to a single dot product — directly expressible as an FMA (Fused Multiply-Add) "
    "SIMD instruction, allowing the compiler to emit AVX2 (8 floats/cycle) or AVX-512 "
    "(16 floats/cycle) instructions."
)
add_code_block(doc,
    "#pragma omp simd reduction(+:dot)\n"
    "for (int d = 0; d < D; ++d)  dot += xi[d] * ck[d];\n"
    "float dist = sqrtf(norm_xi + norm_C[k] - 2.f * dot);"
)

doc.add_page_break()

# ---------------------------------------------------------------------------
# SECTION 4 — SYSTEM ARCHITECTURE
# ---------------------------------------------------------------------------
styled_heading(doc, "4. System Architecture", 1)
add_body(doc,
    "The proposed solver organizes execution into four cache-aligned hardware stages per "
    "iteration. Each stage is designed to target a specific hardware bottleneck:"
)

make_table(doc,
    ["Stage", "Name", "Data Structures", "Parallel Mechanism"],
    [
        ["1", "Aligned Initialization",
         "X[N×D], C[K×D] via 64-byte alloc\nnorm_X[N], norm_C[K]",
         "Eliminates heap fragmentation;\nprepares AVX register boundaries"],
        ["2", "Drift & Bound Maintenance",
         "drift[K], S[K×K]=256KB\nmin_half_dist[K], u[N]=400KB",
         "Compact O(N) upper bounds\nreplace O(N×K) lower bounds"],
        ["3", "Pruned Guided Assignment",
         "Shared reads: X, C, S\nLock-free writes: labels[i], u[i]",
         "schedule(guided,64);\nLevel 1+2 filters skip ~85% evals"],
        ["4", "L1-Resident Reduction",
         "CSR cluster_members[K]\nStack: float local_sum[512]=2KB",
         "Parallel over K clusters;\nzero atomic locks; pure L1 bandwidth"],
    ],
    col_widths=[0.6, 1.5, 2.2, 2.3]
)

styled_heading(doc, "4.1 Stage 1 — 64-Byte Aligned Memory Initialization", 2)
add_body(doc,
    "All primary data arrays are allocated via _aligned_malloc(64, ...) as contiguous flat 1D "
    "buffers with 64-byte cache-line alignment. This eliminates pointer-chasing fragmentation "
    "from nested std::vector structures and ensures AVX2/AVX-512 register loads operate on "
    "naturally aligned memory addresses. Vector norms ‖xᵢ‖² and ‖Cₖ‖² are precomputed once "
    "to avoid redundant inner-loop norm evaluations."
)

styled_heading(doc, "4.2 Stage 2 — Drift and Bound Maintenance", 2)
add_body(doc,
    "At the start of each iteration t: (1) compute per-centroid drift for all k ∈ [K]; "
    "(2) compute global max drift δ_max; (3) compute the K×K pairwise half-distance matrix "
    "S[a][b] using cached centroid norms (256 KB for K=256); (4) compute per-cluster minimum "
    "half-distance mhd[k] used in Level 1 filter. Memory budget: S[K×K] = 256 KB, "
    "u[N] = 400 KB, drift[K] = 1 KB — all fit in L2/L3."
)

styled_heading(doc, "4.3 Stage 3 — Pruned Guided Assignment", 2)
add_body(doc,
    "The assignment loop executes under schedule(guided, 64). For each vector xᵢ: update "
    "upper bound u[i]; apply Level 1 filter (skip all K centroids if satisfied); recompute "
    "tight distance to current centroid via FMA dot-product; for each remaining candidate, "
    "apply Level 2 filter. Final write labels[i] = best_k is lock-free since each i has "
    "an independent memory location. Guided scheduling dispatches exponentially-decaying "
    "chunk sizes so all threads complete simultaneously despite irregular pruning costs."
)

styled_heading(doc, "4.4 Stage 4 — L1-Resident Cluster Reduction", 2)
add_body(doc,
    "Rather than accumulating into shared centroid buffers during assignment, Stage 3 only "
    "writes an integer label. Stage 4 inverts the reduction: (1) build a CSR cluster index "
    "grouping vector IDs by assigned cluster; (2) parallelize over K clusters with "
    "schedule(dynamic, 1); (3) each thread declares a 2 KB stack-local accumulator. "
    "Because 2 KB fits in each core's 32 KB L1 data cache, accumulation runs at peak "
    "L1 bandwidth with zero atomic locks."
)
add_code_block(doc,
    "#pragma omp parallel for schedule(dynamic, 1)\n"
    "for (int k = 0; k < K; ++k) {\n"
    "    float local_sum[512] = {};   // 2 KB — fits in L1\n"
    "    for (int idx : cluster_members[k]) {\n"
    "        const float* xi = X + idx * D;\n"
    "        for (int d = 0; d < D; ++d) local_sum[d] += xi[d];\n"
    "    }\n"
    "    // Write centroid — no lock needed\n"
    "    float inv = 1.f / cluster_members[k].size();\n"
    "    for (int d = 0; d < D; ++d) C[k*D+d] = local_sum[d] * inv;\n"
    "}"
)

add_figure(doc, "c6", "Figure 1: Four-stage cache-aware execution pipeline. All four stages execute within a single iteration loop. Stage 2 through Stage 4 repeat until convergence.", 6.0)

doc.add_page_break()

# ---------------------------------------------------------------------------
# SECTION 5 — MEMORY FOOTPRINT
# ---------------------------------------------------------------------------
styled_heading(doc, "5. Memory Footprint Analysis", 1)
add_body(doc,
    "The table below compares memory requirements of competing approaches for N = 100,000, "
    "K = 256, D = 512, T = 24 threads:"
)
make_table(doc,
    ["Data Structure", "Algorithm", "Size", "Cache Resident?"],
    [
        ["Thread-private T×K×D buffers", "Naive OpenMP",   "8.38 MB",    "❌ Spills to RAM"],
        ["Yinyang lower bounds N×K",      "Yinyang",        "102.4 MB",   "❌ Never fits"],
        ["Upper bounds u[N]",             "Proposed",       "0.38 MB",    "✅ L2/L3"],
        ["Pairwise S[K×K]",               "Proposed",       "0.25 MB",    "✅ L2"],
        ["L1 stack accumulator D floats", "Proposed",       "2 KB/thread","✅ L1 (32 KB)"],
    ],
    col_widths=[2.2, 1.6, 1.0, 1.8]
)
add_figure(doc, "i4",
    "Figure 2: Memory footprint comparison (log scale). The proposed approach replaces "
    "102 MB of Yinyang lower bounds with 0.38 MB scalar upper bounds, and replaces "
    "8.38 MB thread-private buffers with 2 KB per-thread L1-resident stack accumulators.",
    5.8)

# ---------------------------------------------------------------------------
# SECTION 6 — IMPLEMENTATION
# ---------------------------------------------------------------------------
styled_heading(doc, "6. Implementation", 1)

styled_heading(doc, "6.1 Code Architecture", 2)
add_code_block(doc,
    "include/\n"
    "├── common/types.hpp           # alloc64, PerfStats, data generators\n"
    "├── baseline/\n"
    "│   ├── seq_baseline.hpp       # Solver 1: Sequential brute-force\n"
    "│   └── naive_openmp.hpp       # Solver 2: Naive OpenMP + critical\n"
    "└── improved/\n"
    "    ├── pruned_static.hpp      # Solver 3: Pruning + static schedule\n"
    "    ├── pruned_guided.hpp      # Solver 4: Pruning + guided schedule\n"
    "    └── proposed_solver.hpp    # Solver 5: Full cache-aware architecture\n"
    "\n"
    "src/\n"
    "├── baseline/run_baseline.cpp  # Standalone baseline benchmark\n"
    "├── improved/run_improved.cpp  # Standalone improved benchmark\n"
    "└── main.cpp                   # Unified harness (all 5 solvers)"
)

styled_heading(doc, "6.2 Build Configuration", 2)
add_code_block(doc,
    "g++ -O3 -fopenmp -march=native -ffast-math -std=c++17 -I include src/main.cpp -o kmeans_bench"
)
add_body(doc,
    "The -march=native flag enables auto-vectorization targeting the host CPU's AVX2/AVX-512 ISA. "
    "The -ffast-math flag permits the compiler to reorder floating-point operations to maximize "
    "FMA throughput."
)

styled_heading(doc, "6.3 Five Progressive Solver Configurations", 2)
make_table(doc,
    ["Config", "Innovation Added", "Bottleneck Addressed"],
    [
        ["S1 Sequential",         "Brute-force baseline",                  "— (reference)"],
        ["S2 Naive OpenMP",       "schedule(static) + omp critical",       "— (introduces thrashing)"],
        ["S3 + Pruning",          "Two-level triangle inequality filter",   "B1 partial, B2 partial"],
        ["S4 + Guided",           "schedule(guided, 64)",                   "B2 load imbalance"],
        ["S5 Proposed",           "2 KB L1-stack reduction + SIMD",         "B1 cache, B3 contention"],
    ],
    col_widths=[1.4, 2.6, 2.6]
)

doc.add_page_break()

# ---------------------------------------------------------------------------
# SECTION 7 — EXPERIMENTS
# ---------------------------------------------------------------------------
styled_heading(doc, "7. Experimental Setup", 1)
make_table(doc,
    ["Parameter", "Value"],
    [
        ["Number of vectors N",      "100,000"],
        ["Embedding dimension D",    "512"],
        ["Number of clusters K",     "256"],
        ["True Gaussian components", "64"],
        ["Iterations per solver",    "15"],
        ["CPU cores available",      "24"],
        ["Compiler",                 "GCC 16.1.0 (MSYS2)"],
        ["Compile flags",            "-O3 -fopenmp -march=native -ffast-math"],
    ],
    col_widths=[2.8, 3.8]
)
add_body(doc,
    "Synthetic data was generated by sampling N vectors from K_true = 64 isotropic Gaussian "
    "cluster centers, with centroids drawn uniformly from [-5, 5]^D and per-component noise σ = 0.5. "
    "Initial centroids were selected by random sampling of distinct data points. All five solvers "
    "used identical input data and initial centroids to ensure fair comparison."
)

doc.add_page_break()

# ---------------------------------------------------------------------------
# SECTION 8 — RESULTS
# ---------------------------------------------------------------------------
styled_heading(doc, "8. Results and Analysis", 1)

styled_heading(doc, "8.1 Aggregate Performance Summary", 2)
add_figure(doc, "c1",
    "Figure 3: Four-panel benchmark summary. (A) Execution time — proposed achieves 0.62 s "
    "vs 16.52 s sequential. (B) Speedup — 26.86× over baseline. (C) Active distance "
    "evaluations — geometric pruning reduces to 14.8% (85% skipped). (D) Cache miss rate "
    "— L1-tiling drops from 38.4% to 3.2%.",
    6.2)

add_body(doc, "Complete results across all five configurations:")
make_table(doc,
    ["Solver", "Time (s)", "Speedup", "Active Evals%", "Cache Miss%"],
    [
        ["S1: Sequential Baseline",              "16.52", "1.00×",  "100.0%", "12.1%"],
        ["S2: Naive OpenMP (Static+Critical)",   "2.72",  "6.07×",  "100.0%", "38.4%"],
        ["S3: + Geometric Pruning (Static)",     "0.76",  "21.68×", "14.8%",  "36.8%"],
        ["S4: + Guided Scheduling",              "0.74",  "22.41×", "14.8%",  "35.9%"],
        ["S5: Proposed (Full Architecture)",     "0.62",  "26.86×", "14.8%",  "3.2%"],
    ],
    col_widths=[2.8, 0.9, 0.9, 1.1, 1.0],
    header_color="1A5276"
)

styled_heading(doc, "8.2 Cumulative Speedup Waterfall", 2)
add_figure(doc, "c2",
    "Figure 4: Cumulative speedup waterfall. Each bar adds exactly one architectural "
    "optimization. The largest single gain (+15.61×) comes from geometric pruning (S2→S3). "
    "Guided scheduling contributes +0.73×, and L1-tiling adds the final +4.45×.",
    5.8)
add_body(doc,
    "Three key observations: (1) Geometric pruning dominates — the jump from S2 (6.07×) to "
    "S3 (21.68×) is a 3.57× multiplicative improvement from skipping 85% of evaluations. "
    "(2) Naive parallelism is worse than sequential in cache terms — S2 achieves only 6.07× "
    "despite 24 cores due to critical-section serialization and 8.38 MB buffer thrashing. "
    "(3) L1-tiling unlocks the remaining parallelism — S4→S5 gains 4.45× purely from "
    "eliminating cache misses during centroid update."
)

styled_heading(doc, "8.3 Baseline Analysis — The Cache Thrashing Problem", 2)
add_figure(doc, "b3",
    "Figure 5: Cache miss comparison between baseline solvers. Naive OpenMP triples the "
    "cache miss rate from 12.1% to 38.4%, demonstrating that unconstrained thread-private "
    "buffer allocation causes active L2/L3 cache thrashing.",
    5.0)
add_figure(doc, "b1",
    "Figure 6: Per-iteration wall-clock time for baseline solvers. The sequential baseline "
    "takes ~1.1 s per iteration; naive OpenMP reduces this to ~0.18 s — a 6× gain that is "
    "far below the 24× theoretically available from 24 cores.",
    5.8)
add_figure(doc, "b2",
    "Figure 7: Inertia convergence for baseline solvers. Both solvers converge to "
    "approximately the same inertia value, confirming the critical-section update does "
    "not introduce numerical divergence — only performance degradation.",
    5.8)

doc.add_page_break()

styled_heading(doc, "8.4 Geometric Pruning Efficiency", 2)
add_figure(doc, "i1",
    "Figure 8: Distance evaluation pruning percentage per iteration for improved solvers. "
    "The pruning ratio grows with each iteration as centroids stabilize, reaching ~90% "
    "in later iterations. The first iteration has near-zero pruning as bounds are not yet tight.",
    5.8)
add_body(doc,
    "The two-level filter achieves an average 85.2% pruning rate across 15 iterations. "
    "This aligns with observations in Ball k-Means and Yinyang, but with dramatically lower "
    "memory overhead (0.63 MB vs 102 MB). The Level 1 ball filter alone rejects most "
    "interior vectors in O(1) scalar operations, before any O(D) vector computation is initiated. "
    "The two-level filter is mathematically exact — it never discards a point's true nearest "
    "centroid — so the final clustering quality is identical to brute-force Lloyd's algorithm."
)

styled_heading(doc, "8.5 L1-Tiling Effect: Isolating Cache Improvement", 2)
add_figure(doc, "i3",
    "Figure 9: Isolating the L1-tiling contribution. Configurations S4 and S5 share identical "
    "pruning and guided scheduling. L1-tiling drops cache misses by 91% (35.9% → 3.2%) and "
    "reduces total execution time by 16% (0.74 s → 0.62 s).",
    5.8)

styled_heading(doc, "8.6 Per-Iteration Convergence: Improved Solvers", 2)
add_figure(doc, "i2",
    "Figure 10: Per-iteration wall-clock time for improved solvers. Early iterations are slower "
    "as upper bounds are loose and fewer points are pruned. As centroids stabilize, pruning "
    "efficiency increases and later iterations execute progressively faster.",
    5.8)

styled_heading(doc, "8.7 All-Solver Convergence Quality", 2)
add_figure(doc, "c4",
    "Figure 11: Inertia convergence across all five solvers. All configurations converge "
    "to essentially the same final inertia value, confirming the geometric pruning filters "
    "are mathematically exact and do not compromise clustering quality.",
    5.8)

doc.add_page_break()

styled_heading(doc, "8.8 Multi-Core Thread Scaling", 2)
add_figure(doc, "c3",
    "Figure 12: Thread-count scaling for the proposed solver vs ideal linear speedup. "
    "The solver scales efficiently from 1→8 threads (near-ideal), then shows diminishing "
    "returns at higher thread counts due to memory bandwidth saturation at K=256 cluster "
    "centroids shared across cores.",
    5.8)

styled_heading(doc, "8.9 Cache Miss Rate vs. Speedup Correlation", 2)
add_figure(doc, "c5",
    "Figure 13: Scatter plot of cache miss rate vs speedup across all five solver "
    "configurations. A clear negative correlation exists — lower cache miss rates "
    "consistently correspond to higher speedups. The proposed solver occupies the "
    "optimal bottom-right quadrant.",
    5.0)

styled_heading(doc, "8.10 Multi-Metric Radar Comparison", 2)
add_figure(doc, "c7",
    "Figure 14: Multi-metric radar chart normalizing four performance dimensions across "
    "all five solver configurations. The proposed solver (green, outermost) dominates on "
    "all four axes simultaneously — confirming the architectural innovations are "
    "complementary, not competing.",
    4.5)

doc.add_page_break()

# ---------------------------------------------------------------------------
# SECTION 9 — DISCUSSION
# ---------------------------------------------------------------------------
styled_heading(doc, "9. Discussion", 1)

styled_heading(doc, "9.1 Why Naive OpenMP Performs Poorly", 2)
add_body(doc,
    "A common misconception is that adding OpenMP parallelism to a sequential algorithm will "
    "yield near-linear speedup. The naive OpenMP implementation (S2) achieves only 6.07× "
    "speedup on 24 cores — a parallel efficiency of just 25%. The root causes are: "
    "(1) Cache thrashing — the 8.38 MB thread-private accumulator model overflows per-core "
    "L2 and saturates the shared L3 bus; and (2) critical-section serialization — the centroid "
    "update with #pragma omp critical forces all 24 threads to queue sequentially, effectively "
    "eliminating the update phase's parallelism entirely."
)
add_callout(doc,
    "Key insight: Naive parallelism without cache awareness can be WORSE than a carefully "
    "tuned sequential implementation in memory-bandwidth-bound workloads.", "warn")

styled_heading(doc, "9.2 Algorithmic vs Hardware Optimization Tradeoffs", 2)
add_body(doc,
    "The results show that algorithmic optimization (geometric pruning, S2→S3: +15.61× gain) "
    "contributes more than hardware-level cache tuning (L1-tiling, S4→S5: +4.45× gain) in "
    "absolute terms. However, the two approaches target orthogonal bottlenecks: pruning "
    "reduces FLOPs (floating-point operations), while cache-tiling increases effective "
    "memory bandwidth utilization. Together they are multiplicative: "
    "21.68× × 1.24 ≈ 26.9× — consistent with the measured 26.86×."
)

styled_heading(doc, "9.3 Scalability to Production Workloads", 2)
add_body(doc,
    "For real FAISS-scale workloads (N ≥ 10⁶, K = 1024, D = 768): the pairwise matrix "
    "S[K×K] = 1024² × 4 ≈ 4 MB remains within L3 cache; the O(N) upper bound array scales "
    "linearly with dataset size; the 2 KB L1 accumulator is independent of N (depends only "
    "on D); and the cluster CSR index construction is O(N) — a minor overhead. The architecture "
    "is therefore designed to scale gracefully to production embedding dimensions without "
    "memory footprint blowup."
)

# ---------------------------------------------------------------------------
# SECTION 10 — CONCLUSION
# ---------------------------------------------------------------------------
styled_heading(doc, "10. Conclusion", 1)
add_body(doc,
    "This paper presented a hardware-aware, cache-tiled, and lock-free parallel k-means "
    "architecture for high-dimensional AI embeddings. The proposed system resolves three "
    "fundamental bottlenecks in standard parallel k-means: L2/L3 cache overflow from large "
    "thread-private buffers, thread load imbalance from irregular geometric pruning, and "
    "global contention from shared centroid accumulation."
)
add_body(doc,
    "The four-stage architecture — 64-byte aligned initialization, drift-based bound "
    "maintenance, guided-scheduled pruned assignment, and L1-resident cluster reduction — "
    "achieves a 26.86× speedup over the sequential baseline on a 24-core commodity CPU, "
    "while reducing the cache miss rate from 38.4% to 3.2% and skipping 85% of all distance "
    "evaluations via the two-level triangle inequality filter. Importantly, all five solver "
    "configurations converge to identical clustering quality, confirming the correctness "
    "of the lossless geometric filters."
)
add_callout(doc,
    "Future work: Extending the architecture to GPU execution (replacing the L1 accumulator "
    "with shared memory tile strategies), and integrating the solver as a drop-in replacement "
    "for FAISS's flat IVF training component.", "note")

add_horizontal_rule(doc)

# ---------------------------------------------------------------------------
# REFERENCES
# ---------------------------------------------------------------------------
styled_heading(doc, "References", 1)
references = [
    "[1] S. Pei, R. Huang, Y. Hu, Z. Lin, X. Jiang, and Z. Zheng, \"Angle k-Means,\" "
    "Proc. ICLR, 2026.",
    "[2] T. Zhou, Q. Wang, R. Li, S. Mei, S. Yin, R. Hao, and J. Liu, \"Optimizing Yinyang "
    "K-means algorithm on many-core CPUs,\" Journal of National University of Defense "
    "Technology, vol. 46, no. 1, 2024.",
    "[3] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, "
    "L. Hosseini, and H. Jégou, \"The Faiss Library,\" arXiv preprint arXiv:2401.08281, 2024–2026.",
    "[4] S. Xia, D. Peng, D. Meng, C. Zhang, G. Wang, E. Giem, W. Wei, and Z. Chen, \"Ball "
    "k-means: Fast adaptive clustering with no bounds,\" IEEE Trans. Pattern Anal. Mach. "
    "Intell. (TPAMI), vol. 44, no. 1, pp. 87–99, 2022.",
]
for ref in references:
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(ref)
    r.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(5)

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
doc.save(OUT)
print(f"\n✅ Word document saved:\n   {OUT}")
