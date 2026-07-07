#pragma once

/**
 * @file scalar_math.cuh
 * @brief Type-generic scalar math wrappers used by the small math headers.
 *
 * These wrappers keep `f32` and `f64` overloads explicit while allowing vector,
 * matrix, and rotation code to stay templated on `T`.
 */

#include "spr_global_include.h"
#include "config.hpp"

#include <type_traits>

#ifndef SPR_MATH_UNROLL_MAX
#define SPR_MATH_UNROLL_MAX 16
#endif

#ifndef SPR_MATH_SMALL_MAT_MUL_UNROLL_MAX
#define SPR_MATH_SMALL_MAT_MUL_UNROLL_MAX 256
#endif

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE constexpr T sprEpsilon();

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE constexpr f32 sprEpsilon<f32>() { return SPR_EPSILON; }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE constexpr f64 sprEpsilon<f64>() { return F64_EPS; }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprAbs(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprAbs<f32>(f32 x) { return fabsf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprAbs<f64>(f64 x) { return fabs(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprSin(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprSin<f32>(f32 x) { return sprSinf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprSin<f64>(f64 x) { return sin(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprCos(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprCos<f32>(f32 x) { return sprCosf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprCos<f64>(f64 x) { return cos(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE void sprSincos(T x, T* sinOut, T* cosOut);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE void sprSincos<f32>(f32 x, f32* sinOut, f32* cosOut) {
    sprSincosf(x, sinOut, cosOut);
}

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE void sprSincos<f64>(f64 x, f64* sinOut, f64* cosOut) {
#if defined(__CUDA_ARCH__)
    sincos(x, sinOut, cosOut);
#else
    *sinOut = sin(x);
    *cosOut = cos(x);
#endif
}

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprAcos(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprAcos<f32>(f32 x) { return sprAcosf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprAcos<f64>(f64 x) { return acos(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprAtan2(T y, T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprAtan2<f32>(f32 y, f32 x) { return sprAtan2f(y, x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprAtan2<f64>(f64 y, f64 x) { return atan2(y, x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprExp(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprExp<f32>(f32 x) { return sprExpf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprExp<f64>(f64 x) { return exp(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprLog(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprLog<f32>(f32 x) { return sprLogf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprLog<f64>(f64 x) { return log(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprSqrt(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprSqrt<f32>(f32 x) { return sprSqrtf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprSqrt<f64>(f64 x) { return sqrt(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprRsqrt(T x);

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f32 sprRsqrt<f32>(f32 x) { return sprRsqrtf(x); }

template<>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE f64 sprRsqrt<f64>(f64 x) { return 1.0 / sqrt(x); }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprMin(T a, T b) { return a < b ? a : b; }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprMax(T a, T b) { return a > b ? a : b; }

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE T sprClamp(T x, T low, T high) {
    return sprMin(sprMax(x, low), high);
}

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE bool sprIsFinite(T x) {
    return isfinite(x);
}

template<class T>
SPR_CUDA_HOST_DEVICE_FORCE_INLINE bool sprNearlyEqual(T a, T b, T eps = sprEpsilon<T>()) {
    return sprAbs(a - b) <= eps;
}

SPR_CUDA_HOST_DEVICE_INLINE constexpr std::size_t sprMathNextPowerOfTwo(std::size_t value) {
    if (value <= 1) {
        return 1;
    }

    --value;
    for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
        value |= value >> shift;
    }
    return value + 1;
}

/** Alignment policy for fixed-size vectors, capped for CUDA-friendly storage. */
template<i32 N, typename T>
struct VecAlign {
    static constexpr std::size_t storageAlign = sprMathNextPowerOfTwo(sizeof(T) * std::size_t(N));
    static constexpr std::size_t cappedAlign = storageAlign > 32 ? 32 : storageAlign;
    static constexpr std::size_t minAlign = cappedAlign < alignof(T) ? alignof(T) : cappedAlign;
    static constexpr std::size_t value = sprMathNextPowerOfTwo(minAlign);

    static_assert((value % alignof(T)) == 0, "Alignment must be multiple of alignof(T)");
    static_assert((value & (value - 1)) == 0, "Alignment must be power-of-two");
};
