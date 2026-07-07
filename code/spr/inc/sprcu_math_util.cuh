#pragma once

/**
 * @file sprcu_math_util.cuh
 * @brief Small scalar combinatorics and interpolation helpers.
 *
 * These utilities are intentionally simple, inline, and CUDA-callable. Integer
 * combinatorics use `u64` arithmetic and wrap on overflow.
 */

#include "config.hpp"
#include "spr_global_include.h"

#include <type_traits>

// These integer helpers use u64 arithmetic and wrap on overflow.
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr u64 factorial(u64 n, u64 low = 2) {
    if (low > n) {
        return 1;
    }

    u64 result = 1;
    for (u64 i = low; i <= n; ++i) {
        result *= i;
    }
    return result;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr u64 choose(u64 n, u64 k) {
    if (k > n) {
        return 0;
    }

    if (k > n - k) {
        k = n - k;
    }

    u64 result = 1;
    for (u64 i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}

// these strange template versions are for cases where you want to use these computations as, say, template parameters 
// unsure if they're worth keeping around! we'll see
template<u64 N, u64 D>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr u64 factorial() {
    if constexpr (N > D) {
        return N * factorial<N - 1, D>();
    } else {
        return 1;
    }
}

template<u64 N>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr u64 factorial() {
    return factorial<N, 1>();
}

template<u64 N, u64 K>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr u64 choose() {
    static_assert(K <= N, "choose<N, K> requires K <= N");

    if constexpr (K > N - K) {
        return factorial<N, K>() / factorial<N - K>();
    } else {
        return factorial<N, N - K>() / factorial<K>();
    }
}

template<typename T, typename S = f32>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr T lerp(const T& a, const T& b, S t) {
    return a + (b - a) * t;
}

/** Rank a sorted-or-unsorted symmetric tensor multi-index in packed storage. */
template <typename... Ts>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
constexpr u64 symRank(Ts... indices) {
    static_assert(sizeof...(Ts) > 0, "symRank requires at least one index");
    static_assert((std::is_integral_v<Ts> && ...), "symRank indices must be integral");

    if constexpr (config::MATH_SAFETY) {
        const bool hasNegative = ((std::is_signed_v<Ts> && indices < 0) || ...);
        SPR_ASSERT(!hasNegative && "symRank indices must be non-negative");
        if (hasNegative) {
            return 0;
        }
    }

    constexpr u64 N = sizeof...(Ts);
    u64 arr[N] = { static_cast<u64>(indices)... };

    for (u64 i = 1; i < N; ++i) {
        u64 key = arr[i];
        u64 j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            --j;
        }
        arr[j] = key;
    }

    u64 rank = 0;
    for (u64 i = 0; i < N; ++i) {
        rank += choose(arr[i] + i, i + 1);
    }

    return rank;
}
