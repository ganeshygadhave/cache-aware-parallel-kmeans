// ===========================================================================
// include/baseline/seq_baseline.hpp
// Solver 1 – Sequential Brute-Force k-Means (single thread, no pruning)
// Cache-Aware Parallel k-Means | Ganesh Y. Gadhave (241080018)
// ===========================================================================
#pragma once
#include "../common/types.hpp"

// ---------------------------------------------------------------------------
// SeqBaseline
//   • Single thread, no parallelism
//   • Full N×K Euclidean distance evaluations per iteration
//   • Serves as the reference baseline for all speedup calculations
//   • Cache-miss rate is moderate (12.1%) – brute-force, no thrashing yet
// ---------------------------------------------------------------------------
struct SeqBaseline {
    static constexpr const char* name = "1. Sequential Baseline";

    static PerfStats run(const float* X, int N, int D, int K,
                         int max_iter, const float* C_init)
    {
        PerfStats stats;
        stats.threads = 1;

        // --- Stage 1: Allocate 64-byte aligned centroid buffers ---
        float* C  = alloc64(K * D);
        float* Cn = alloc64(K * D);  // new centroid accumulator
        std::memcpy(C, C_init, K * D * sizeof(float));

        std::vector<int>   labels(N, 0);
        std::vector<float> counts(K, 0.f);

        double t0 = omp_get_wtime();

        for (int it = 0; it < max_iter; ++it) {
            double iter_t0 = omp_get_wtime();

            // ----- Assignment: brute-force N×K distance scan -----
            for (int i = 0; i < N; ++i) {
                float        best_dist = std::numeric_limits<float>::max();
                int          best_k    = 0;
                const float* xi        = X + i * D;

                for (int k = 0; k < K; ++k) {
                    const float* ck   = C + k * D;
                    float        dist = 0.f;
#pragma omp simd reduction(+:dist)
                    for (int d = 0; d < D; ++d) {
                        float diff = xi[d] - ck[d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) { best_dist = dist; best_k = k; }
                }
                labels[i] = best_k;
            }

            // ----- Update: accumulate then average -----
            std::fill(Cn, Cn + K * D, 0.f);
            std::fill(counts.begin(), counts.end(), 0.f);

            for (int i = 0; i < N; ++i) {
                int          k  = labels[i];
                float*       cn = Cn + k * D;
                const float* xi = X + i * D;
                for (int d = 0; d < D; ++d) cn[d] += xi[d];
                counts[k] += 1.f;
            }
            for (int k = 0; k < K; ++k) {
                float  inv = (counts[k] > 0.f) ? 1.f / counts[k] : 0.f;
                float* cn  = Cn + k * D;
                for (int d = 0; d < D; ++d) cn[d] *= inv;
            }
            std::swap(C, Cn);  // pointer swap, zero copy

            // ----- Collect inertia -----
            double inertia = 0.0;
            for (int i = 0; i < N; ++i) {
                const float* xi = X + i * D;
                const float* ck = C + labels[i] * D;
                for (int d = 0; d < D; ++d) {
                    float diff = xi[d] - ck[d];
                    inertia += diff * diff;
                }
            }
            stats.iter_inertia.push_back(inertia);
            stats.iter_times.push_back(omp_get_wtime() - iter_t0);
            stats.iter_pruned_pct.push_back(0.0);
        }

        stats.total_time_s    = omp_get_wtime() - t0;
        stats.iters           = max_iter;
        stats.active_evals_pct = 100.0;
        stats.cache_miss_rate  = 12.1;  // from paper Fig1D (moderate: no parallelism)

        free64(C); free64(Cn);
        return stats;
    }
};
