// ===========================================================================
// include/common/types.hpp
// Shared types, aligned allocators, PerfStats, and data generator
// Cache-Aware Parallel k-Means | Ganesh Y. Gadhave (241080018)
// ===========================================================================
#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>
#include <omp.h>

// ---------------------------------------------------------------------------
// 64-byte aligned allocation (matches AVX-512 cache-line width)
// ---------------------------------------------------------------------------
static inline float* alloc64(std::size_t n) {
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(n * sizeof(float), 64);
#else
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    return static_cast<float*>(p);
}

static inline void free64(float* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// ---------------------------------------------------------------------------
// PerfStats – collected by every solver and consumed by main + plotter
// ---------------------------------------------------------------------------
struct PerfStats {
    double total_time_s     = 0.0;   // total wall-clock time
    double speedup          = 1.0;   // vs sequential baseline
    double active_evals_pct = 100.0; // % distance evaluations NOT pruned
    double cache_miss_rate  = 0.0;   // simulated / reported value
    int    iters            = 0;
    int    threads          = 1;

    // Per-iteration data for convergence / time plots
    std::vector<double> iter_times;
    std::vector<double> iter_inertia;
    std::vector<double> iter_pruned_pct; // % evaluations pruned per iteration
};

// ---------------------------------------------------------------------------
// Scaling result (thread-count sweep)
// ---------------------------------------------------------------------------
struct ScalingResult {
    int    threads;
    double time_s;
    double speedup;
};

// ---------------------------------------------------------------------------
// Synthetic D-dimensional Gaussian mixture embedding generator
// ---------------------------------------------------------------------------
inline void generate_embeddings(float* X, int N, int D, int K_true,
                                 uint32_t seed = 0)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float>      normal(0.f, 1.f);
    std::uniform_int_distribution<int>   which(0, K_true - 1);
    std::uniform_real_distribution<float> ctr(-5.f, 5.f);

    std::vector<float> centers(K_true * D);
    for (auto& v : centers) v = ctr(rng);

    for (int i = 0; i < N; ++i) {
        int          c  = which(rng);
        float*       xi = X + i * D;
        const float* cc = centers.data() + c * D;
        for (int d = 0; d < D; ++d) xi[d] = cc[d] + 0.5f * normal(rng);
    }
}

// ---------------------------------------------------------------------------
// k-means++ style initialisation – pick K shuffled data points
// ---------------------------------------------------------------------------
inline void init_centroids(float* C_init, const float* X,
                            int N, int D, int K, uint32_t seed = 1)
{
    std::mt19937 rng(seed);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int k = 0; k < K; ++k)
        std::memcpy(C_init + k * D, X + idx[k] * D, D * sizeof(float));
}
