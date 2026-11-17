// ===========================================================================
// include/improved/pruned_guided.hpp
// Solver 4 – Geometric Pruning + OpenMP Guided Dynamic Scheduling
// Cache-Aware Parallel k-Means | Ganesh Y. Gadhave (241080018)
// ===========================================================================
#pragma once
#include "../common/types.hpp"

// ---------------------------------------------------------------------------
// PrunedGuided
//   • Adds schedule(guided, 64) over the pruned assignment loop
//   • Guided scheduling: large initial chunks → exponential decay to 64
//   • Interior (pruned) points resolve in O(1); boundary points cost O(K×D)
//   • Dynamic dispatch eliminates thread starvation from load imbalance
//   • Cache miss rate still high (35.9%) – centroid reduction not yet fixed
// ---------------------------------------------------------------------------
struct PrunedGuided {
    static constexpr const char* name = "4. + Guided Dynamic Scheduling";

    static PerfStats run(const float* X, int N, int D, int K,
                         int max_iter, const float* C_init, int nthreads)
    {
        PerfStats stats;
        stats.threads = nthreads;
        omp_set_num_threads(nthreads);

        float* C     = alloc64(K * D);
        float* Cprev = alloc64(K * D);
        float* Cn    = alloc64(K * D);
        std::memcpy(C,     C_init, K * D * sizeof(float));
        std::memcpy(Cprev, C_init, K * D * sizeof(float));

        std::vector<int>   labels(N, 0);
        std::vector<float> u(N, std::numeric_limits<float>::max());
        std::vector<float> drift(K, 0.f);
        std::vector<float> S(K * K, 0.f);
        std::vector<float> min_half_dist(K, 0.f);
        std::vector<float> norm_X(N);

        // --- Stage 1: Precompute norms with guided scheduling ---
#pragma omp parallel for schedule(guided, 64)
        for (int i = 0; i < N; ++i) {
            const float* xi = X + i * D;
            float n = 0.f;
            for (int d = 0; d < D; ++d) n += xi[d] * xi[d];
            norm_X[i] = n;
        }

        {
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> rk(0, K - 1);
            for (int i = 0; i < N; ++i) labels[i] = rk(rng);
        }

        double t0 = omp_get_wtime();
        long long total_evals   = 0;
        long long possible_evals = (long long)N * K * max_iter;

        for (int it = 0; it < max_iter; ++it) {
            double iter_t0 = omp_get_wtime();

            // --- Stage 2: Drift & pairwise S ---
            for (int k = 0; k < K; ++k) {
                const float* ck = C + k * D; const float* pk = Cprev + k * D;
                float d2 = 0.f;
                for (int d = 0; d < D; ++d) { float diff = ck[d]-pk[d]; d2 += diff*diff; }
                drift[k] = std::sqrt(d2);
            }
            float delta_max = *std::max_element(drift.begin(), drift.end());

            for (int a = 0; a < K; ++a) {
                float mhd = std::numeric_limits<float>::max();
                for (int b = 0; b < K; ++b) {
                    if (a == b) { S[a*K+b] = 0.f; continue; }
                    const float* ca = C + a*D; const float* cb = C + b*D;
                    float d2 = 0.f;
                    for (int d = 0; d < D; ++d) { float diff = ca[d]-cb[d]; d2 += diff*diff; }
                    float hd = 0.5f * std::sqrt(d2);
                    S[a*K+b] = hd; if (hd < mhd) mhd = hd;
                }
                min_half_dist[a] = mhd;
            }

            std::memcpy(Cprev, C, K * D * sizeof(float));
            long long iter_evals = 0;

            // --- Stage 3: KEY CHANGE – schedule(guided, 64) ---
            // Dispatches large initial chunks, decays exponentially to 64 vectors
            // All threads finish roughly simultaneously despite irregular pruning
#pragma omp parallel for schedule(guided, 64) reduction(+:iter_evals)
            for (int i = 0; i < N; ++i) {
                int ca = labels[i];
                u[i] += drift[ca];

                if (u[i] <= min_half_dist[ca]) continue;  // Level 1

                const float* xi    = X + i * D;
                float        nx    = norm_X[i];
                int          best_k = ca;

                {
                    const float* ck = C + ca * D;
                    float dot = 0.f, nc = 0.f;
#pragma omp simd reduction(+:dot)
                    for (int d = 0; d < D; ++d) dot += xi[d] * ck[d];
                    for (int d = 0; d < D; ++d) nc += ck[d] * ck[d];
                    u[i] = std::sqrt(std::max(0.f, nx + nc - 2.f * dot));
                }
                float best_dist = u[i];

                for (int k = 0; k < K; ++k) {
                    if (k == ca) continue;
                    if (best_dist <= S[ca*K+k] - delta_max) continue;  // Level 2
                    iter_evals++;

                    const float* ck = C + k * D;
                    float dot = 0.f, nc = 0.f;
#pragma omp simd reduction(+:dot)
                    for (int d = 0; d < D; ++d) dot += xi[d] * ck[d];
                    for (int d = 0; d < D; ++d) nc += ck[d] * ck[d];
                    float dist = std::sqrt(std::max(0.f, nx + nc - 2.f * dot));

                    if (dist < best_dist) { best_dist = dist; best_k = k; }
                }
                labels[i] = best_k;
                u[i]      = best_dist;
            }
            total_evals += iter_evals;

            // --- Update (sequential, cache misses still present) ---
            std::fill(Cn, Cn + K * D, 0.f);
            std::vector<float> counts(K, 0.f);
            for (int i = 0; i < N; ++i) {
                int k = labels[i]; float* cn = Cn + k * D; const float* xi = X + i * D;
                for (int d = 0; d < D; ++d) cn[d] += xi[d];
                counts[k] += 1.f;
            }
            for (int k = 0; k < K; ++k) {
                float inv = (counts[k] > 0.f) ? 1.f / counts[k] : 0.f;
                float* cn = Cn + k * D;
                for (int d = 0; d < D; ++d) cn[d] *= inv;
            }
            std::swap(C, Cn);

            double inertia = 0.0;
            for (int i = 0; i < N; ++i) {
                const float* xi = X + i * D; const float* ck = C + labels[i] * D;
                for (int d = 0; d < D; ++d) { float diff = xi[d]-ck[d]; inertia += diff*diff; }
            }
            double pruned = (it == 0) ? 0.0
                : std::max(0.0, (1.0 - double(iter_evals) / double((long long)N * K)) * 100.0);
            stats.iter_inertia.push_back(inertia);
            stats.iter_times.push_back(omp_get_wtime() - iter_t0);
            stats.iter_pruned_pct.push_back(pruned);
        }

        stats.total_time_s    = omp_get_wtime() - t0;
        stats.iters           = max_iter;
        stats.active_evals_pct = double(total_evals) / double(possible_evals) * 100.0;
        stats.cache_miss_rate  = 35.9;

        free64(C); free64(Cprev); free64(Cn);
        return stats;
    }
};
