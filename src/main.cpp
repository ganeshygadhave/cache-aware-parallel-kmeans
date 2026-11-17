// ===========================================================================
// src/main.cpp
// Combined Benchmark Harness – runs ALL 5 solvers and writes unified CSVs
// Cache-Aware Parallel k-Means | Ganesh Y. Gadhave (241080018)
// Course : Parallel Computing [Batch A]
//
// Build (from project root):
//   g++ -O3 -fopenmp -march=native -ffast-math -std=c++17 \
//       -I include src/main.cpp -o kmeans_bench
// ===========================================================================

// --- Baseline solvers ---
#include "../include/baseline/seq_baseline.hpp"
#include "../include/baseline/naive_openmp.hpp"

// --- Improved solvers ---
#include "../include/improved/pruned_static.hpp"
#include "../include/improved/pruned_guided.hpp"
#include "../include/improved/proposed_solver.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------
static constexpr int N        = 100'000;   // number of vectors
static constexpr int D        = 512;       // embedding dimensionality
static constexpr int K        = 256;       // number of clusters
static constexpr int K_TRUE   = 64;        // true Gaussian mixture components
static constexpr int MAX_ITER = 15;        // iterations per solver
static constexpr int SEED     = 42;

// ---------------------------------------------------------------------------
// Pretty printer
// ---------------------------------------------------------------------------
static void print_header() {
    std::cout << "\n"
              << "================================================================\n"
              << "  Cache-Aware Parallel k-Means – Full Benchmark\n"
              << "  N=" << N << "  D=" << D << "  K=" << K
              << "  Iters=" << MAX_ITER << "\n"
              << "================================================================\n"
              << std::left
              << std::setw(44) << "Solver"
              << std::setw(12) << "Time (s)"
              << std::setw(10) << "Speedup"
              << std::setw(16) << "Active Evals%"
              << std::setw(14) << "Cache Miss%"
              << "\n"
              << std::string(96, '-') << "\n";
}

static void print_row(const char* name, const PerfStats& s) {
    std::cout << std::left
              << std::setw(44) << name
              << std::setw(12) << std::fixed << std::setprecision(2) << s.total_time_s
              << std::setw(10) << std::setprecision(2) << s.speedup
              << std::setw(16) << std::setprecision(1) << s.active_evals_pct
              << std::setw(14) << std::setprecision(1) << s.cache_miss_rate
              << "\n";
}

// ---------------------------------------------------------------------------
// CSV writers – unified files consumed by plot_results.py
// ---------------------------------------------------------------------------
static void write_summary_csv(const std::vector<PerfStats>& r,
                               const std::vector<std::string>& n)
{
    std::ofstream f("benchmark_summary.csv");
    f << "solver,time_s,speedup,active_evals_pct,cache_miss_rate\n";
    for (std::size_t i = 0; i < r.size(); ++i)
        f << "\"" << n[i] << "\","
          << std::fixed << std::setprecision(3)
          << r[i].total_time_s << "," << r[i].speedup << ","
          << r[i].active_evals_pct << "," << r[i].cache_miss_rate << "\n";
    std::cout << "[CSV] benchmark_summary.csv\n";
}

static void write_iter_csv(const std::vector<PerfStats>& r,
                            const std::vector<std::string>& n)
{
    std::ofstream f("benchmark_iterations.csv");
    f << "solver,iteration,iter_time_s,inertia,pruned_pct\n";
    for (std::size_t i = 0; i < r.size(); ++i)
        for (int it = 0; it < (int)r[i].iter_inertia.size(); ++it)
            f << "\"" << n[i] << "\"," << (it + 1) << ","
              << std::fixed << std::setprecision(4)
              << r[i].iter_times[it] << ","
              << r[i].iter_inertia[it] << ","
              << r[i].iter_pruned_pct[it] << "\n";
    std::cout << "[CSV] benchmark_iterations.csv\n";
}

static void write_scaling_csv(const std::vector<ScalingResult>& s)
{
    std::ofstream f("benchmark_scaling.csv");
    f << "threads,time_s,speedup\n";
    for (const auto& r : s)
        f << r.threads << ","
          << std::fixed << std::setprecision(3)
          << r.time_s << "," << r.speedup << "\n";
    std::cout << "[CSV] benchmark_scaling.csv\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP max threads: " << max_threads << "\n";

    // ---- Stage 1: Allocate and generate embeddings ----
    std::cout << "\n[Data] Generating " << N << " × D=" << D << " embeddings ("
              << (N * D * sizeof(float) / 1024 / 1024) << " MB)...\n";

    float* X      = alloc64((std::size_t)N * D);
    float* C_init = alloc64(K * D);
    generate_embeddings(X, N, D, K_TRUE, SEED);
    init_centroids(C_init, X, N, D, K, SEED + 1);

    // ---- Run all solvers ----
    print_header();
    std::vector<PerfStats>   results;
    std::vector<std::string> names;
    double seq_time = 0.0;

    // ---- BASELINE ----
    std::cout << "\n--- BASELINE SOLVERS ---\n";

    std::cout << "[1/5] Sequential Baseline...\n";
    auto s1 = SeqBaseline::run(X, N, D, K, MAX_ITER, C_init);
    s1.speedup = 1.0;  seq_time = s1.total_time_s;
    print_row(SeqBaseline::name, s1);
    results.push_back(s1); names.push_back(SeqBaseline::name);

    std::cout << "[2/5] Naive OpenMP (Static + Critical)...\n";
    auto s2 = NaiveOpenMP::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    s2.speedup = seq_time / s2.total_time_s;
    print_row(NaiveOpenMP::name, s2);
    results.push_back(s2); names.push_back(NaiveOpenMP::name);

    // ---- IMPROVED ----
    std::cout << "\n--- IMPROVED SOLVERS ---\n";

    std::cout << "[3/5] + Low-Memory Geometric Pruning (Static)...\n";
    auto s3 = PrunedStatic::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    s3.speedup = seq_time / s3.total_time_s;
    print_row(PrunedStatic::name, s3);
    results.push_back(s3); names.push_back(PrunedStatic::name);

    std::cout << "[4/5] + Guided Dynamic Scheduling...\n";
    auto s4 = PrunedGuided::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    s4.speedup = seq_time / s4.total_time_s;
    print_row(PrunedGuided::name, s4);
    results.push_back(s4); names.push_back(PrunedGuided::name);

    std::cout << "[5/5] Proposed (L1-Tiling + SIMD + Guided)...\n";
    auto s5 = ProposedSolver::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    s5.speedup = seq_time / s5.total_time_s;
    print_row(ProposedSolver::name, s5);
    results.push_back(s5); names.push_back(ProposedSolver::name);

    // ---- Thread-scaling sweep ----
    std::cout << "\n--- THREAD SCALING (Proposed solver) ---\n";
    std::vector<ScalingResult> scaling;

    auto tmp1 = ProposedSolver::run(X, N, D, K, MAX_ITER, C_init, 1);
    double base = tmp1.total_time_s;
    scaling.push_back({1, base, 1.0});
    std::cout << "  T=1   " << std::fixed << std::setprecision(3) << base << "s  1.00x\n";

    int thread_counts[] = {2, 4, 6, 8, 10, 12, 16, 20, 24};
    for (int t : thread_counts) {
        if (t > max_threads) break;
        auto tmp = ProposedSolver::run(X, N, D, K, MAX_ITER, C_init, t);
        double sp = base / tmp.total_time_s;
        scaling.push_back({t, tmp.total_time_s, sp});
        std::cout << "  T=" << std::setw(3) << t << " "
                  << tmp.total_time_s << "s  "
                  << std::setprecision(2) << sp << "x\n";
    }

    // ---- Write CSVs ----
    std::cout << "\n[Output] Writing CSV files...\n";
    write_summary_csv(results, names);
    write_iter_csv(results, names);
    write_scaling_csv(scaling);

    // ---- Final table ----
    std::cout << "\n";
    print_header();
    for (std::size_t i = 0; i < results.size(); ++i)
        print_row(names[i].c_str(), results[i]);

    std::cout << "\n[Done] Run: python plot_results.py\n";

    free64(X); free64(C_init);
    return 0;
}
