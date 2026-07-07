#pragma once

/**
 * @file affine.cuh
 * @brief Affine transforms: linear matrix plus translation.
 *
 * `Affine<T, N>` represents `x' = A*x + t`. Normal transformation uses
 * inverse-transpose semantics, which requires the linear part to be invertible.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "vec.cuh"
#include "mat.cuh"
#include "rot.cuh"

/** Affine transform `x' = A*x + t`. */
template<class T, u32 N>
struct Affine {
    Mat<T, N, N> A;
    Vec<T, N> t;

    SPR_CUDA_HOST_DEVICE_INLINE Affine() : A(Mat<T, N, N>::eye()), t(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE Affine(Mat<T, N, N> A_, Vec<T, N> t_) : A(A_), t(t_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Affine identity() { return Affine(); }
};

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyPoint(Affine<T, N> A, Vec<T, N> p) {
    return A.A * p + A.t;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyVector(Affine<T, N> A, Vec<T, N> v) {
    return A.A * v;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> applyNormal(Affine<T, N> A, Vec<T, N> n) {
    // For nonuniform scale/shear, normals transform by A^{-T}.
    // inverse(mat4) is intentionally not included in this scaffold yet.
    return transpose(inverse(A.A)) * n;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Affine<T, N> operator*(Affine<T, N> A, Affine<T, N> B) {
    return Affine<T, N>(A.A * B.A, A.A * B.t + A.t);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Affine<T, N> inverse(Affine<T, N> A) {
    Mat<T, N, N> Ai = inverse(A.A);
    return Affine<T, N>(Ai, -(Ai * A.t));
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, N + 1, N + 1> homogeneous(Affine<T, N> A) {
    Mat<T, N + 1, N + 1> out;

    SPR_UNROLL
    for (u32 r = 0; r < N; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < N; ++c) { out[r][c] = A.A[r][c]; }
        out[r][N] = A.t[r];
    }

    out[N][N] = T(1);
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Affine<T, N> affineFromRotTranslation(Rot<T, N> R, Vec<T, N> t) {
    return Affine<T, N>(matrix(R), t);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Affine<T, 3> affineFromTrs(Vec<T, 3> translation, Rot<T, 3> rotation, Vec<T, 3> scale) {
    Mat<T, 3, 3> A = matrix(rotation);

    // A = R * diag(scale): scale columns under column-vector convention.
    SPR_UNROLL
    for (u32 r = 0; r < 3; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < 3; ++c) { A[r][c] *= scale[c]; }
    }

    return Affine<T, 3>(A, translation);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, 4, 4> composeTrs(Vec<T, 3> translation, Rot<T, 3> rotation, Vec<T, 3> scale = Vec<T, 3>(T(1))) {
    return homogeneous(affineFromTrs(translation, rotation, scale));
}

template<u32 N>
using affine = Affine<f32, N>;

using affine2 = Affine<f32, 2>;
using affine3 = Affine<f32, 3>;
using affine4 = Affine<f32, 4>;

using daffine2 = Affine<f64, 2>;
using daffine3 = Affine<f64, 3>;
using daffine4 = Affine<f64, 4>;
