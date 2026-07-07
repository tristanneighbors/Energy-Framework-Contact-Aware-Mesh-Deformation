#pragma once

/**
 * @file geometry_util.cuh
 * @brief Small geometry helpers built on vectors and rotations.
 *
 * These are convenience functions for common formula-level operations. In math
 * safety mode, degenerate projection, angle, and look-at inputs assert and
 * return conservative fallback values.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "vec.cuh"
#include "mat.cuh"
#include "quat.cuh"
#include "rot.cuh"

/** Project `v` onto `onto`. Degenerate `onto` returns zero in safety mode. */
template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> project(Vec<T, N> v, Vec<T, N> onto) {
    const T denom = sqNorm(onto);
    if constexpr (config::MATH_SAFETY) {
        const bool valid = (denom > sprEpsilon<T>()) && sprIsFinite(denom);
        SPR_ASSERT(valid && "project(v, onto) requires nonzero finite 'onto' vector");
        if (!valid) { return Vec<T, N>::zero(); }
    }
    return (dot(v, onto) / denom) * onto;
}

/** Angle between two vectors in radians. Degenerate inputs return zero in safety mode. */
template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T angleBetween(Vec<T, N> v, Vec<T, N> w) {
    const T lv2 = sqNorm(v);
    const T lw2 = sqNorm(w);
    if constexpr (config::MATH_SAFETY) {
        const bool valid = (lv2 > sprEpsilon<T>()) && (lw2 > sprEpsilon<T>()) && sprIsFinite(lv2) && sprIsFinite(lw2);
        SPR_ASSERT(valid && "angleBetween requires nonzero finite vectors");
        if (!valid) { return T(0); }
    }

    const T c = dot(v, w) * sprRsqrt<T>(lv2 * lw2);
    return sprAcos<T>(sprClamp<T>(c, T(-1), T(1)));
}

// Convenience short name.
template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T angle(Vec<T, N> v, Vec<T, N> w) {
    return angleBetween(v, w);
}

/** Camera-style look-at orientation with identity fallback for collapsed bases. */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> lookAtRot3(Vec<T, 3> pos, Vec<T, 3> target, Vec<T, 3> up) {
    if constexpr (config::MATH_SAFETY) {
        const T upNorm2 = sqNorm(up);
        if (!(upNorm2 > sprEpsilon<T>()) || !sprIsFinite(upNorm2)) {
            return Rot<T, 3>::identity();
        }
    }
    normalize(up);

    Vec<T, 3> yp = pos - target;
    if constexpr (config::MATH_SAFETY) {
        const T viewNorm2 = sqNorm(yp);
        if (!(viewNorm2 > sprEpsilon<T>()) || !sprIsFinite(viewNorm2)) {
            return Rot<T, 3>::identity();
        }
    }
    normalize(yp);

    Vec<T, 3> xp = cross(yp, up);
    if constexpr (config::MATH_SAFETY) {
        const T rightNorm2 = sqNorm(xp);
        if (!(rightNorm2 > sprEpsilon<T>()) || !sprIsFinite(rightNorm2)) {
            return Rot<T, 3>::identity();
        }
    }
    normalize(xp);

    Vec<T, 3> zp = cross(xp, yp);
    normalize(zp);

    Quat<T> q;
    const T trace = xp.x + yp.y + zp.z;

    if (trace > T(0)) {
        const T S = sprSqrt<T>(trace + T(1)) * T(2);
        q.re = T(0.25) * S;
        q.imx = (zp.y - yp.z) / S;
        q.imy = (xp.z - zp.x) / S;
        q.imz = (yp.x - xp.y) / S;
    } else if ((xp.x > yp.y) && (xp.x > zp.z)) {
        const T S = sprSqrt<T>(T(1) + xp.x - yp.y - zp.z) * T(2);
        q.re = (zp.y - yp.z) / S;
        q.imx = T(0.25) * S;
        q.imy = (xp.y + yp.x) / S;
        q.imz = (xp.z + zp.x) / S;
    } else if (yp.y > zp.z) {
        const T S = sprSqrt<T>(T(1) + yp.y - xp.x - zp.z) * T(2);
        q.re = (xp.z - zp.x) / S;
        q.imx = (xp.y + yp.x) / S;
        q.imy = T(0.25) * S;
        q.imz = (yp.z + zp.y) / S;
    } else {
        const T S = sprSqrt<T>(T(1) + zp.z - xp.x - yp.y) * T(2);
        q.re = (yp.x - xp.y) / S;
        q.imx = (xp.z + zp.x) / S;
        q.imy = (yp.z + zp.y) / S;
        q.imz = T(0.25) * S;
    }

    normalize(q);
    return rot3FromUnitQuatUnchecked(q);
}
