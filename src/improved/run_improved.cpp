

#include "../../include/improved/pruned_static.hpp"
#include "../../include/improved/pruned_guided.hpp"
#include "../../include/improved/proposed_solver.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>


static constexpr int N        = 100'000;
static constexpr int D        = 512;
static constexpr int K        = 256;
static constexpr int K_TRUE   = 64;
static constexpr int MAX_ITER = 15;
static constexpr int SEED     = 42;


static double SEQ_REF_TIME = 0.0;

static void write_csv(const std::vector<PerfStats>& results,
                      const std::vector<std::string>& names,
                      const std::vector<ScalingResult>& scaling)
{
    // Summary CSV
    {
        std::ofstream f("improved_summary.csv");
        f << "solver,time_s,speedup,active_evals_pct,cache_miss_rate\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            f << "\"" << names[i] << "\","
              << std::fixed << std::setprecision(3)
              << r.total_time_s << "," << r.speedup << ","
              << r.active_evals_pct << "," << r.cache_miss_rate << "\n";
        }
        std::cout << "[CSV] improved_summary.csv written\n";
    }
    // Iteration CSV
    {
        std::ofstream f("improved_iterations.csv");
        f << "solver,iteration,iter_time_s,inertia,pruned_pct\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            for (int it = 0; it < (int)r.iter_inertia.size(); ++it) {
                f << "\"" << names[i] << "\","
                  << (it + 1) << ","
                  << std::fixed << std::setprecision(4)
                  << r.iter_times[it] << ","
                  << r.iter_inertia[it] << ","
                  << r.iter_pruned_pct[it] << "\n";
            }
        }
        std::cout << "[CSV] improved_iterations.csv written\n";
    }
    // Scaling CSV
    {
        std::ofstream f("improved_scaling.csv");
        f << "threads,time_s,speedup\n";
        for (const auto& r : scaling) {
            f << r.threads << ","
              << std::fixed << std::setprecision(3)
              << r.time_s << "," << r.speedup << "\n";
        }
        std::cout << "[CSV] improved_scaling.csv written\n";
    }
}

int main() {
    int max_threads = omp_get_max_threads();
    std::cout << "\n=== IMPROVED SOLVER BENCHMARK ===\n"
              << "N=" << N << "  D=" << D << "  K=" << K
              << "  Threads=" << max_threads << "\n\n";

    float* X      = alloc64((std::size_t)N * D);
    float* C_init = alloc64(K * D);
    generate_embeddings(X, N, D, K_TRUE, SEED);
    init_centroids(C_init, X, N, D, K, SEED + 1);

    std::vector<PerfStats>   results;
    std::vector<std::string> names;

    std::cout << "[1/3] Geometric Pruning (Static schedule)...\n";
    auto s3 = PrunedStatic::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    if (SEQ_REF_TIME > 0) s3.speedup = SEQ_REF_TIME / s3.total_time_s;
    std::cout << "      Time=" << std::fixed << std::setprecision(3)
              << s3.total_time_s << "s  ActiveEvals="
              << std::setprecision(1) << s3.active_evals_pct << "%\n";
    results.push_back(s3); names.push_back(PrunedStatic::name);

    std::cout << "[2/3] + Guided Dynamic Scheduling...\n";
    auto s4 = PrunedGuided::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    if (SEQ_REF_TIME > 0) s4.speedup = SEQ_REF_TIME / s4.total_time_s;
    std::cout << "      Time=" << s4.total_time_s << "s\n";
    results.push_back(s4); names.push_back(PrunedGuided::name);

    std::cout << "[3/3] Proposed (Full: L1-Tiling + SIMD + Guided)...\n";
    auto s5 = ProposedSolver::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    if (SEQ_REF_TIME > 0) s5.speedup = SEQ_REF_TIME / s5.total_time_s;
    std::cout << "      Time=" << s5.total_time_s << "s  CacheMiss="
              << s5.cache_miss_rate << "%\n";
    results.push_back(s5); names.push_back(ProposedSolver::name);

    // ---- Thread-scaling sweep (Proposed solver) ----
    std::cout << "\n[Scaling] Proposed solver thread-count sweep...\n";
    std::vector<ScalingResult> scaling;

    auto tmp1 = ProposedSolver::run(X, N, D, K, MAX_ITER, C_init, 1);
    double base_time = tmp1.total_time_s;
    scaling.push_back({1, base_time, 1.0});
    std::cout << "  T=1  " << std::fixed << std::setprecision(3)
              << base_time << "s  1.00x\n";

    int thread_counts[] = {2, 4, 6, 8, 10, 12, 16, 20, 24};
    for (int t : thread_counts) {
        if (t > max_threads) break;
        auto tmp = ProposedSolver::run(X, N, D, K, MAX_ITER, C_init, t);
        double sp = base_time / tmp.total_time_s;
        scaling.push_back({t, tmp.total_time_s, sp});
        std::cout << "  T=" << t << "  " << tmp.total_time_s << "s  "
                  << std::setprecision(2) << sp << "x\n";
    }

    write_csv(results, names, scaling);

    free64(X); free64(C_init);
    std::cout << "\n[Done] Improved benchmark complete.\n";
    return 0;
}
