#pragma once

/**
 * @file vec.cuh
 * @brief Fixed-size arithmetic vectors and basic vector operations.
 *
 * `Vec<T, N>` is the general small vector type. Dimensions 2, 3, and 4 add
 * named component access while keeping the same arithmetic model. Safety-mode
 * normalization maps degenerate vectors to zero; with math safety disabled,
 * callers are responsible for avoiding zero, non-finite, and otherwise
 * degenerate inputs.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "scalar_math.cuh"

#include <type_traits>

/** Fixed-size arithmetic vector intended for small CUDA-friendly dimensions. */
template<class T, u32 N>
struct alignas(VecAlign<N, T>::value) Vec {
    static_assert(N > 0, "Vec<T, N> requires N > 0");
    static_assert(std::is_arithmetic_v<T>, "Vec<T, N> requires arithmetic scalar T");

    T data[N];

    SPR_CUDA_HOST_DEVICE_INLINE Vec() {
        SPR_UNROLL
        for (u32 i = 0; i < N; ++i) { data[i] = T(0); }
    }

    SPR_CUDA_HOST_DEVICE_INLINE explicit Vec(T a) {
        SPR_UNROLL
        for (u32 i = 0; i < N; ++i) { data[i] = a; }
    }

    SPR_CUDA_HOST_DEVICE_INLINE T& operator[](u32 idx) { return data[idx]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator[](u32 idx) const { return data[idx]; }

    SPR_CUDA_HOST_DEVICE_INLINE static Vec zero() { return Vec(T(0)); }
};

template<class T>
struct alignas(VecAlign<2, T>::value) Vec<T, 2> {
    static_assert(std::is_arithmetic_v<T>, "Vec<T, 2> requires arithmetic scalar T");

    union {
        struct {
            union { T x; T u; };
            union { T y; T v; };
        };
        T data[2];
    };

    SPR_CUDA_HOST_DEVICE_INLINE Vec() : x(T(0)), y(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Vec(T a) : x(a), y(a) {}
    SPR_CUDA_HOST_DEVICE_INLINE Vec(T x_, T y_) : x(x_), y(y_) {}

    SPR_CUDA_HOST_DEVICE_INLINE T& operator[](u32 idx) { return data[idx]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator[](u32 idx) const { return data[idx]; }

    SPR_CUDA_HOST_DEVICE_INLINE static Vec zero() { return Vec(T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec xhat() { return Vec(T(1), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec yhat() { return Vec(T(0), T(1)); }
};

template<class T>
struct alignas(VecAlign<3, T>::value) Vec<T, 3> {
    static_assert(std::is_arithmetic_v<T>, "Vec<T, 3> requires arithmetic scalar T");

    union {
        struct {
            union { T x; T r; };
            union { T y; T g; };
            union { T z; T b; };
        };
        T data[3];
    };

    SPR_CUDA_HOST_DEVICE_INLINE Vec() : x(T(0)), y(T(0)), z(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Vec(T a) : x(a), y(a), z(a) {}
    SPR_CUDA_HOST_DEVICE_INLINE Vec(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    SPR_CUDA_HOST_DEVICE_INLINE T& operator[](u32 idx) { return data[idx]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator[](u32 idx) const { return data[idx]; }

    SPR_CUDA_HOST_DEVICE_INLINE static Vec zero() { return Vec(T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec xhat() { return Vec(T(1), T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec yhat() { return Vec(T(0), T(1), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec zhat() { return Vec(T(0), T(0), T(1)); }
};

template<class T>
struct alignas(VecAlign<4, T>::value) Vec<T, 4> {
    static_assert(std::is_arithmetic_v<T>, "Vec<T, 4> requires arithmetic scalar T");

    union {
        struct {
            union { T x; T r; };
            union { T y; T g; };
            union { T z; T b; };
            union { T w; T a; };
        };
        T data[4];
    };

    SPR_CUDA_HOST_DEVICE_INLINE Vec() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Vec(T s) : x(s), y(s), z(s), w(s) {}
    SPR_CUDA_HOST_DEVICE_INLINE Vec(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

    SPR_CUDA_HOST_DEVICE_INLINE T& operator[](u32 idx) { return data[idx]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator[](u32 idx) const { return data[idx]; }

    SPR_CUDA_HOST_DEVICE_INLINE static Vec zero() { return Vec(T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec xhat() { return Vec(T(1), T(0), T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec yhat() { return Vec(T(0), T(1), T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec zhat() { return Vec(T(0), T(0), T(1), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Vec what() { return Vec(T(0), T(0), T(0), T(1)); }
};

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator+(const Vec<T, N>& a, const Vec<T, N>& b) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = a[i] + b[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N>& operator+=(Vec<T, N>& a, const Vec<T, N>& b) {
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { a[i] += b[i]; }
    return a;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator-(const Vec<T, N>& a, const Vec<T, N>& b) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = a[i] - b[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N>& operator-=(Vec<T, N>& a, const Vec<T, N>& b) {
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { a[i] -= b[i]; }
    return a;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator-(const Vec<T, N>& v) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = -v[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(const Vec<T, N>& v, T scalar) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = v[i] * scalar; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(T scalar, const Vec<T, N>& v) { return v * scalar; }

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N>& operator*=(Vec<T, N>& v, T scalar) {
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { v[i] *= scalar; }
    return v;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator/(const Vec<T, N>& v, T scalar) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = v[i] / scalar; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N>& operator/=(Vec<T, N>& v, T scalar) {
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { v[i] /= scalar; }
    return v;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(const Vec<T, N>& a, const Vec<T, N>& b) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = a[i] * b[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N>& operator*=(Vec<T, N>& a, const Vec<T, N>& b) {
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { a[i] *= b[i]; }
    return a;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator/(const Vec<T, N>& a, const Vec<T, N>& b) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i] = a[i] / b[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N>& operator/=(Vec<T, N>& a, const Vec<T, N>& b) {
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { a[i] /= b[i]; }
    return a;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T sqNorm(const Vec<T, N>& v) {
    T sum = T(0);
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { sum += v[i] * v[i]; }
    return sum;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T length(const Vec<T, N>& v) {
    return sprSqrt<T>(sqNorm(v));
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T dot(const Vec<T, N>& a, const Vec<T, N>& b) {
    T sum = T(0);
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { sum += a[i] * b[i]; }
    return sum;
}


template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE bool isUnit(const Vec<T, N>& v, T eps = T(8) * sprEpsilon<T>()) {
    return sprAbs(sqNorm(v) - T(1)) <= eps;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE void normalize(Vec<T, N>& v) {
    const T lsq = sqNorm(v);
    if constexpr (config::MATH_SAFETY) {
        if (!(lsq > sprEpsilon<T>()) || !sprIsFinite(lsq)) {
            SPR_UNROLL
            for (u32 i = 0; i < N; ++i) { v[i] = T(0); }
            return;
        }
    }
    v *= sprRsqrt<T>(lsq);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> normal(const Vec<T, N>& v) {
    Vec<T, N> out = v;
    normalize(out);
    return out;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T cross(const Vec<T, 2>& a, const Vec<T, 2>& b) {
    return a.x * b.y - a.y * b.x;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 3> cross(const Vec<T, 3>& a, const Vec<T, 3>& b) {
    return Vec<T, 3>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

template<u32 N>
using vec = Vec<f32, N>;

using vec2 = Vec<f32, 2>;
using vec3 = Vec<f32, 3>;
using vec4 = Vec<f32, 4>;

using dvec2 = Vec<f64, 2>;
using dvec3 = Vec<f64, 3>;
using dvec4 = Vec<f64, 4>;
