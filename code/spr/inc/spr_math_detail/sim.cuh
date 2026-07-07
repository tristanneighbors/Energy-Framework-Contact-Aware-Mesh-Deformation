#pragma once

/**
 * @file sim.cuh
 * @brief Similarity transforms: uniform scale, rotation, and translation.
 *
 * `Sim<T, N>` represents `x' = s*R*x + t`. Normals are returned with rotational
 * direction only; callers that need scaled normal magnitudes should handle that
 * policy explicitly.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "vec.cuh"
#include "mat.cuh"
#include "rot.cuh"

/** Similarity transform `x' = s*R*x + t`. */
template<class T, u32 N>
struct Sim {
    Rot<T, N> R;
    Vec<T, N> t;
    T s;

    SPR_CUDA_HOST_DEVICE_INLINE Sim() : R(), t(T(0)), s(T(1)) {}
    SPR_CUDA_HOST_DEVICE_INLINE Sim(Rot<T, N> R_, Vec<T, N> t_, T s_) : R(R_), t(t_), s(s_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Sim identity() { return Sim(); }
};

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyPoint(Sim<T, N> S, Vec<T, N> p) {
    return S.s * (S.R * p) + S.t;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyVector(Sim<T, N> S, Vec<T, N> v) {
    return S.s * (S.R * v);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyNormal(Sim<T, N> S, Vec<T, N> n) {
    // Uniform scale only changes normal magnitude, so the direction is R*n.
    // If users need inverse-scale normal magnitude, add a separate function.
    return S.R * n;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sim<T, N> operator*(Sim<T, N> A, Sim<T, N> B) {
    return Sim<T, N>(A.R * B.R, A.s * (A.R * B.t) + A.t, A.s * B.s);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Sim<T, N> inverse(Sim<T, N> S) {
    Rot<T, N> Ri = inverse(S.R);
    T si = T(1) / S.s;
    return Sim<T, N>(Ri, si * (Ri * (-S.t)), si);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, N + 1, N + 1> homogeneous(Sim<T, N> S) {
    Mat<T, N + 1, N + 1> out;
    Mat<T, N, N> Rm = matrix(S.R);

    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < N; ++c) { out[r][c] = S.s * Rm[r][c]; }
        out[r][N] = S.t[r];
    }

    out[N][N] = T(1);
    return out;
}

template<u32 N>
using sim = Sim<f32, N>;

using sim2 = Sim<f32, 2>;
using sim3 = Sim<f32, 3>;
using sim4 = Sim<f32, 4>;

using dsim2 = Sim<f64, 2>;
using dsim3 = Sim<f64, 3>;
using dsim4 = Sim<f64, 4>;
