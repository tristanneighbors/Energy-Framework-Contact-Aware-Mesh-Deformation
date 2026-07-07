#pragma once

/**
 * @file sym.cuh
 * @brief Packed symmetric matrices.
 *
 * `Sym<T, N>` stores one triangle of a symmetric matrix while still exposing
 * `(r, c)` indexing. This is meant for small covariance, metric, and quadratic
 * form data where full matrix storage is unnecessary.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "scalar_math.cuh"
#include "vec.cuh"
#include "mat.cuh"

/** Compact symmetric matrix with lower-triangle row-major packed storage. */
template<class T, u32 N>
struct Sym {
    static_assert(N > 0, "Sym<T, N> requires N > 0");
    static constexpr u32 count = (N * (N + 1)) / 2;

    T data[count];

    SPR_CUDA_HOST_DEVICE_INLINE Sym() {
        SPR_UNROLL
        for (u32 i = 0; i < count; ++i) { data[i] = T(0); }
    }

    SPR_CUDA_HOST_DEVICE_INLINE explicit Sym(T diagValue) {
        SPR_UNROLL
        for (u32 i = 0; i < count; ++i) { data[i] = T(0); }
        SPR_UNROLL
        for (u32 i = 0; i < N; ++i) { (*this)(i, i) = diagValue; }
    }

    SPR_CUDA_HOST_DEVICE_INLINE static constexpr u32 idx(u32 r, u32 c) {
        return (r >= c) ? ((r * (r + 1)) / 2 + c) : ((c * (c + 1)) / 2 + r);
    }

    SPR_CUDA_HOST_DEVICE_INLINE T& operator()(u32 r, u32 c) { return data[idx(r, c)]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator()(u32 r, u32 c) const { return data[idx(r, c)]; }

    SPR_CUDA_HOST_DEVICE_INLINE static Sym zero() { return Sym(); }
    SPR_CUDA_HOST_DEVICE_INLINE static Sym identity() { return Sym(T(1)); }
};

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> operator+(const Sym<T, N>& A, const Sym<T, N>& B) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < Sym<T, N>::count; ++i) { out.data[i] = A.data[i] + B.data[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N>& operator+=(Sym<T, N>& A, const Sym<T, N>& B) {
    SPR_UNROLL
    for (u32 i = 0; i < Sym<T, N>::count; ++i) { A.data[i] += B.data[i]; }
    return A;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> operator-(const Sym<T, N>& A, const Sym<T, N>& B) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < Sym<T, N>::count; ++i) { out.data[i] = A.data[i] - B.data[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N>& operator-=(Sym<T, N>& A, const Sym<T, N>& B) {
    SPR_UNROLL
    for (u32 i = 0; i < Sym<T, N>::count; ++i) { A.data[i] -= B.data[i]; }
    return A;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> operator*(const Sym<T, N>& A, T s) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < Sym<T, N>::count; ++i) { out.data[i] = A.data[i] * s; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> operator*(T s, const Sym<T, N>& A) { return A * s; }

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N>& operator*=(Sym<T, N>& A, T s) {
    SPR_UNROLL
    for (u32 i = 0; i < Sym<T, N>::count; ++i) { A.data[i] *= s; }
    return A;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> outerSym(const Vec<T, N>& x) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c <= r; ++c) { out(r, c) = x[r] * x[c]; }
    }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE void addOuterSym(Sym<T, N>& A, const Vec<T, N>& x, T weight = T(1)) {
    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c <= r; ++c) { A(r, c) += weight * x[r] * x[c]; }
    }
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(const Sym<T, N>& A, const Vec<T, N>& x) {
    Vec<T, N> out;
    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        T sum = T(0);
        SPR_UNROLL
        for (u32 c = 0; c < N; ++c) { sum += A(r, c) * x[c]; }
        out[r] = sum;
    }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T quadratic(const Sym<T, N>& A, const Vec<T, N>& x) {
    return dot(x, A * x);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, N, N> matrix(const Sym<T, N>& A) {
    Mat<T, N, N> out;
    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < N; ++c) { out[r][c] = A(r, c); }
    }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> symFromMatrixLower(const Mat<T, N, N>& A) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c <= r; ++c) { out(r, c) = A[r][c]; }
    }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> symFromMatrixAverage(const Mat<T, N, N>& A) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c <= r; ++c) { out(r, c) = T(0.5) * (A[r][c] + A[c][r]); }
    }
    return out;
}

template<u32 N>
using sym = Sym<f32, N>;

using sym2 = Sym<f32, 2>;
using sym3 = Sym<f32, 3>;
using sym4 = Sym<f32, 4>;
using sym6 = Sym<f32, 6>;

using dsym2 = Sym<f64, 2>;
using dsym3 = Sym<f64, 3>;
using dsym4 = Sym<f64, 4>;
using dsym6 = Sym<f64, 6>;
