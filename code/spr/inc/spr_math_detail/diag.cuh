#pragma once

/**
 * @file diag.cuh
 * @brief Diagonal matrix wrapper backed by its diagonal vector.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "vec.cuh"
#include "mat.cuh"
#include "sym.cuh"

/** Diagonal matrix represented by its diagonal entries. */
template<class T, u32 N>
struct Diag {
    Vec<T, N> d;

    SPR_CUDA_HOST_DEVICE_INLINE Diag() : d(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Diag(T a) : d(a) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Diag(Vec<T, N> d_) : d(d_) {}

    SPR_CUDA_HOST_DEVICE_INLINE T& operator[](u32 i) { return d[i]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator[](u32 i) const { return d[i]; }

    SPR_CUDA_HOST_DEVICE_INLINE static Diag zero() { return Diag(T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Diag identity() { return Diag(T(1)); }
};

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(Diag<T, N> D, Vec<T, N> x) {
    return D.d * x;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(Vec<T, N> x, Diag<T, N> D) {
    return x * D.d;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Diag<T, N> operator*(Diag<T, N> D, T s) {
    return Diag<T, N>(D.d * s);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Diag<T, N> operator*(T s, Diag<T, N> D) { return D * s; }

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, N, N> matrix(Diag<T, N> D) {
    Mat<T, N, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out[i][i] = D.d[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sym<T, N> symMatrix(Diag<T, N> D) {
    Sym<T, N> out;
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { out(i, i) = D.d[i]; }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T quadratic(Diag<T, N> D, Vec<T, N> x) {
    T sum = T(0);
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { sum += D.d[i] * x[i] * x[i]; }
    return sum;
}

template<u32 N>
using diag = Diag<f32, N>;

using diag2 = Diag<f32, 2>;
using diag3 = Diag<f32, 3>;
using diag4 = Diag<f32, 4>;

using ddiag2 = Diag<f64, 2>;
using ddiag3 = Diag<f64, 3>;
using ddiag4 = Diag<f64, 4>;
