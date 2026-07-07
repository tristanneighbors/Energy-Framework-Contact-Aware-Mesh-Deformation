#pragma once

/**
 * @file spr_global_include.h
 * @brief Project-wide scalar aliases, limits, CUDA qualifiers, and scalar math wrappers.
 *
 * This is the small common substrate used by the public headers. It deliberately
 * stays flat and dependency-light so CUDA and non-CUDA translation units can
 * share the same scalar names and inline helper macros.
 */

#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#ifdef __cplusplus
  #include <cstddef>
#endif

#ifndef __cplusplus
  #include <stdbool.h>
#endif

#define SPR_STDC_VERSION_C99 199901L
#define SPR_STDC_VERSION_C11 201112L
#define SPR_STDC_VERSION_C17 201710L
#define SPR_STDC_VERSION_C23 202311L

#ifndef STDC_VERSION_C99
  #define STDC_VERSION_C99 SPR_STDC_VERSION_C99
#endif
#ifndef STDC_VERSION_C11
  #define STDC_VERSION_C11 SPR_STDC_VERSION_C11
#endif
#ifndef STDC_VERSION_C17
  #define STDC_VERSION_C17 SPR_STDC_VERSION_C17
#endif
#ifndef STDC_VERSION_C23
  #define STDC_VERSION_C23 SPR_STDC_VERSION_C23
#endif

#if defined(__cplusplus)
  #define SPR_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= SPR_STDC_VERSION_C11)
  #define SPR_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
  #define SPR_STATIC_ASSERT_CONCAT_(a, b) a##b
  #define SPR_STATIC_ASSERT_CONCAT(a, b) SPR_STATIC_ASSERT_CONCAT_(a, b)
  #define SPR_STATIC_ASSERT(cond, msg) typedef char SPR_STATIC_ASSERT_CONCAT(static_assertion_, __LINE__)[(cond) ? 1 : -1]
#endif

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define U8_MAX UINT8_MAX
#define U16_MAX UINT16_MAX
#define U32_MAX UINT32_MAX
#define U64_MAX UINT64_MAX

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

#define I8_MAX INT8_MAX
#define I8_MIN INT8_MIN
#define I16_MAX INT16_MAX
#define I16_MIN INT16_MIN
#define I32_MAX INT32_MAX
#define I32_MIN INT32_MIN
#define I64_MAX INT64_MAX
#define I64_MIN INT64_MIN

typedef float f32;
typedef double f64;

SPR_STATIC_ASSERT(sizeof(f32) == 4, "f32 must be 4 bytes.");
SPR_STATIC_ASSERT(sizeof(f64) == 8, "f64 must be 8 bytes.");

#ifdef __cplusplus
static_assert(sizeof(float) == 4, "float is not 32-bit on this platform");
static_assert(sizeof(double) == 8, "double is not 64-bit on this platform");

#if defined(SPR_REQUIRE_64BIT_PLATFORM)
static_assert(sizeof(std::size_t) == 8, "std::size_t is not 64-bit on this platform");
#endif
#endif

#define F32_MAX FLT_MAX
#define F32_MIN_POS FLT_MIN
#define F32_LOWEST (-FLT_MAX)
#define F64_MAX DBL_MAX
#define F64_MIN_POS DBL_MIN
#define F64_LOWEST (-DBL_MAX)
#define F32_EPS FLT_EPSILON
#define F64_EPS DBL_EPSILON

#if defined(__clang__) || defined(__GNUC__)
  #define SPR_ALWAYS_INLINE inline __attribute__((always_inline))
  #define SPR_NEVER_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
  #define SPR_ALWAYS_INLINE __forceinline
  #define SPR_NEVER_INLINE __declspec(noinline)
#else
  #define SPR_ALWAYS_INLINE inline
  #define SPR_NEVER_INLINE
#endif

#ifndef SPR_ASSERT
  #define SPR_ASSERT(expr) assert(expr)
#endif

#ifdef __CUDACC__
  #define SPR_CUDA_DEVICE __device__
  #define SPR_CUDA_HOST __host__
  #define SPR_CUDA_HOST_DEVICE __host__ __device__
  #define SPR_CUDA_INLINE inline
  #define SPR_CUDA_FORCE_INLINE __forceinline__
#else
  #define SPR_CUDA_DEVICE
  #define SPR_CUDA_HOST
  #define SPR_CUDA_HOST_DEVICE
  #define SPR_CUDA_INLINE inline
  #define SPR_CUDA_FORCE_INLINE inline
#endif

#define SPR_CUDA_HOST_DEVICE_INLINE SPR_CUDA_HOST_DEVICE SPR_CUDA_INLINE
#define SPR_CUDA_HOST_DEVICE_FORCE_INLINE SPR_CUDA_HOST_DEVICE SPR_CUDA_FORCE_INLINE

#ifndef SPR_IDX
  #define SPR_IDX(x, y, width) ((y) * (width) + (x))
#endif

#ifndef SPR_EPSILON
  #define SPR_EPSILON 0.00001f
#endif
#ifndef SPR_PI
  #define SPR_PI 3.141592653589793238462643383279502884
#endif

#if defined(__CUDA_ARCH__)
  #define SPR_UNROLL _Pragma("unroll")
#elif defined(__clang__) || defined(__GNUC__)
  #define SPR_UNROLL _Pragma("unroll")
#else
  #define SPR_UNROLL
#endif

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprSinf(f32 x) {
  return sinf(x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprCosf(f32 x) {
  return cosf(x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void sprSincosf(f32 x, f32 *sinOut, f32 *cosOut) {
#if defined(__CUDA_ARCH__)
  sincosf(x, sinOut, cosOut);
#else
  *sinOut = sinf(x);
  *cosOut = cosf(x);
#endif
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprAcosf(f32 x) {
  return acosf(x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprAtan2f(f32 y, f32 x) {
  return atan2f(y, x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprExpf(f32 x) {
  return expf(x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprLogf(f32 x) {
  return logf(x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprSqrtf(f32 x) {
  return sqrtf(x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprRsqrtf(f32 x) {
#if defined(__CUDA_ARCH__)
  return rsqrtf(x);
#else
  return 1.0f / sqrtf(x);
#endif
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprFmaxf(f32 a, f32 b) {
  return a > b ? a : b;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprFminf(f32 a, f32 b) {
  return a < b ? a : b;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sprClampf(f32 x, f32 low, f32 high) {
  return sprFminf(sprFmaxf(x, low), high);
}
