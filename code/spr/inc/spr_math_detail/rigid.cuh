#pragma once

/**
 * @file rigid.cuh
 * @brief Rigid transforms: rotation plus translation.
 *
 * `Rigid<T, N>` represents `x' = R*x + t`. Points receive the translation;
 * vectors and normals receive only the rotation.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "vec.cuh"
#include "mat.cuh"
#include "rot.cuh"

/** Rigid transform `x' = R*x + t`. */
template<class T, u32 N>
struct Rigid {
    Rot<T, N> R;
    Vec<T, N> t;

    SPR_CUDA_HOST_DEVICE_INLINE Rigid() : R(), t(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE Rigid(Rot<T, N> R_, Vec<T, N> t_) : R(R_), t(t_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Rigid identity() { return Rigid(); }
};

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyPoint(Rigid<T, N> Tform, Vec<T, N> p) {
    return Tform.R * p + Tform.t;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyVector(Rigid<T, N> Tform, Vec<T, N> v) {
    return Tform.R * v;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyNormal(Rigid<T, N> Tform, Vec<T, N> n) {
    return Tform.R * n;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Rigid<T, N> operator*(Rigid<T, N> A, Rigid<T, N> B) {
    return Rigid<T, N>(A.R * B.R, A.R * B.t + A.t);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Rigid<T, N> inverse(Rigid<T, N> Tform) {
    Rot<T, N> Ri = inverse(Tform.R);
    return Rigid<T, N>(Ri, -(Ri * Tform.t));
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, N + 1, N + 1> homogeneous(Rigid<T, N> Tform) {
    Mat<T, N + 1, N + 1> out;
    Mat<T, N, N> Rm = matrix(Tform.R);

    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < N; ++c) { out[r][c] = Rm[r][c]; }
        out[r][N] = Tform.t[r];
    }

    out[N][N] = T(1);
    return out;
}

template<u32 N>
using rigid = Rigid<f32, N>;

using rigid2 = Rigid<f32, 2>;
using rigid3 = Rigid<f32, 3>;
using rigid4 = Rigid<f32, 4>;

using drigid2 = Rigid<f64, 2>;
using drigid3 = Rigid<f64, 3>;
using drigid4 = Rigid<f64, 4>;
