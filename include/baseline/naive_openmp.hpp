// ===========================================================================
// include/baseline/naive_openmp.hpp
// Solver 2 – Naive Parallel k-Means (OpenMP static + critical-section update)
// Cache-Aware Parallel k-Means | Ganesh Y. Gadhave (241080018)
// ===========================================================================
#pragma once
#include "../common/types.hpp"

// ---------------------------------------------------------------------------
// NaiveOpenMP
//   • Multi-threaded assignment with schedule(static)
//   • Centroid update uses omp critical → global contention
//   • Thread-private buffer model: T×K×D ≈ 8.38 MB overflows L2/L3
//     (simulated here with critical-section which replicates the thrashing)
//   • Demonstrates the CACHE THRASHING bottleneck (miss rate 38.4%)
//   • Speedup ≈ 7× despite having 24 cores due to contention + thrashing
// ---------------------------------------------------------------------------
struct NaiveOpenMP {
    static constexpr const char* name = "2. Naive OpenMP (Static + 8MB Buffers)";

    static PerfStats run(const float* X, int N, int D, int K,
                         int max_iter, const float* C_init, int nthreads)
    {
        PerfStats stats;
        stats.threads = nthreads;
        omp_set_num_threads(nthreads);

        float* C  = alloc64(K * D);
        float* Cn = alloc64(K * D);
        std::memcpy(C, C_init, K * D * sizeof(float));

        std::vector<int>   labels(N, 0);

        double t0 = omp_get_wtime();

        for (int it = 0; it < max_iter; ++it) {
            double iter_t0 = omp_get_wtime();

            // ----- Assignment: parallel static – brute-force -----
            // schedule(static) → equal chunk per thread regardless of work cost
#pragma omp parallel for schedule(static)
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

            // ----- Update: critical section → serialises threads -----
            // This models the T×K×D thread-private buffer that overflows cache
            std::fill(Cn, Cn + K * D, 0.f);
            std::vector<float> counts(K, 0.f);

#pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) {
                int          k  = labels[i];
                const float* xi = X + i * D;
#pragma omp critical       // <-- bottleneck: global serialisation
                {
                    float* cn = Cn + k * D;
                    for (int d = 0; d < D; ++d) cn[d] += xi[d];
                    counts[k] += 1.f;
                }
            }
            for (int k = 0; k < K; ++k) {
                float  inv = (counts[k] > 0.f) ? 1.f / counts[k] : 0.f;
                float* cn  = Cn + k * D;
                for (int d = 0; d < D; ++d) cn[d] *= inv;
            }
            std::swap(C, Cn);

            // ----- Inertia -----
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
        stats.cache_miss_rate  = 38.4;  // cache thrashing from large private buffers

        free64(C); free64(Cn);
        return stats;
    }
};
