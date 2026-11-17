
#include "../../include/baseline/seq_baseline.hpp"
#include "../../include/baseline/naive_openmp.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>


static constexpr int N        = 100'000;
static constexpr int D        = 512;
static constexpr int K        = 256;
static constexpr int K_TRUE   = 64;
static constexpr int MAX_ITER = 15;
static constexpr int SEED     = 42;

static void write_csv(const std::vector<PerfStats>& results,
                      const std::vector<std::string>& names)
{
    // Summary CSV
    {
        std::ofstream f("baseline_summary.csv");
        f << "solver,time_s,speedup,active_evals_pct,cache_miss_rate\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            f << "\"" << names[i] << "\","
              << std::fixed << std::setprecision(3)
              << r.total_time_s  << "," << r.speedup << ","
              << r.active_evals_pct << "," << r.cache_miss_rate << "\n";
        }
        std::cout << "[CSV] baseline_summary.csv written\n";
    }
    // Iteration CSV
    {
        std::ofstream f("baseline_iterations.csv");
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
        std::cout << "[CSV] baseline_iterations.csv written\n";
    }
}

int main() {
    int max_threads = omp_get_max_threads();
    std::cout << "\n=== BASELINE BENCHMARK ===\n"
              << "N=" << N << "  D=" << D << "  K=" << K
              << "  Threads=" << max_threads << "\n\n";

    // Data
    float* X      = alloc64((std::size_t)N * D);
    float* C_init = alloc64(K * D);
    generate_embeddings(X, N, D, K_TRUE, SEED);
    init_centroids(C_init, X, N, D, K, SEED + 1);

    std::vector<PerfStats>   results;
    std::vector<std::string> names;
    double seq_time = 0.0;

    std::cout << "[1/2] Sequential Baseline...\n";
    auto s1 = SeqBaseline::run(X, N, D, K, MAX_ITER, C_init);
    s1.speedup = 1.0;
    seq_time   = s1.total_time_s;
    std::cout << "      Time=" << std::fixed << std::setprecision(2)
              << s1.total_time_s << "s  Speedup=1.00x\n";
    results.push_back(s1); names.push_back(SeqBaseline::name);

    std::cout << "[2/2] Naive OpenMP (Static + Critical)...\n";
    auto s2 = NaiveOpenMP::run(X, N, D, K, MAX_ITER, C_init, max_threads);
    s2.speedup = seq_time / s2.total_time_s;
    std::cout << "      Time=" << s2.total_time_s << "s  Speedup="
              << std::setprecision(2) << s2.speedup << "x  "
              << "CacheMiss=" << s2.cache_miss_rate << "%\n";
    results.push_back(s2); names.push_back(NaiveOpenMP::name);

    write_csv(results, names);

    free64(X); free64(C_init);
    std::cout << "\n[Done] Baseline benchmark complete.\n";
    return 0;
}
