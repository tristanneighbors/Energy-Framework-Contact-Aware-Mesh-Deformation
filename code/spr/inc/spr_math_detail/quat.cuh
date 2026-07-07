#pragma once

#include "spr_global_include.h"
#include "config.hpp"
#include "scalar_math.cuh"
#include "vec.cuh"

/**
 * @file quat.cuh
 * @brief Quaternion algebra type and quaternion exponential/logarithm helpers.
 *
 * `Quat<T>` is the generic quaternion algebra type. It is intentionally
 * separate from `Rot<T, 3>`: a quaternion may have any norm and supports
 * algebraic operations such as addition and scalar multiplication, while
 * `Rot<T, 3>` represents an element of SO(3) and stores a unit quaternion as
 * an implementation detail.
 */

/**
 * @brief Quaternion algebra value `re + imx*i + imy*j + imz*k`.
 *
 * @tparam T Scalar type, normally `f32` or `f64`.
 *
 * A `Quat<T>` is not required to be unit length. Use `rot3` / `Rot<T, 3>` for
 * semantic 3D rotations and use `rot3FromQuat`, `rot3FromUnitQuat`, or
 * `rot3FromUnitQuatUnchecked` to cross that boundary explicitly.
 *
 * `Vec<T, 4>` interop uses ordinary vector component order:
 *
 * ```text
 * vec4(x, y, z, w) <-> quat(w, x, y, z)
 * ```
 */
template<class T>
struct alignas(VecAlign<4, T>::value) Quat {
    /** Real scalar component. */
    T re;
    /** Coefficient of the `i` basis element. */
    T imx;
    /** Coefficient of the `j` basis element. */
    T imy;
    /** Coefficient of the `k` basis element. */
    T imz;

    SPR_CUDA_HOST_DEVICE_INLINE Quat() : re(T(0)), imx(T(0)), imy(T(0)), imz(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Quat(T a) : re(a), imx(T(0)), imy(T(0)), imz(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE Quat(T re_, T imx_, T imy_, T imz_) : re(re_), imx(imx_), imy(imy_), imz(imz_) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Quat(Vec<T, 3> v) : re(T(0)), imx(v.x), imy(v.y), imz(v.z) {}

    /** Build from `vec4(x, y, z, w)` as `quat(w, x, y, z)`. */
    SPR_CUDA_HOST_DEVICE_INLINE explicit Quat(const Vec<T, 4>& v) : re(v.w), imx(v.x), imy(v.y), imz(v.z) {}
    /** Convert to `vec4(imx, imy, imz, re)`. */
    SPR_CUDA_HOST_DEVICE_INLINE explicit operator Vec<T, 4>() const { return Vec<T, 4>(imx, imy, imz, re); }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator+(const Quat& q) const { return Quat(re + q.re, imx + q.imx, imy + q.imy, imz + q.imz); }
    SPR_CUDA_HOST_DEVICE_INLINE Quat& operator+=(Quat q) { re += q.re; imx += q.imx; imy += q.imy; imz += q.imz; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator-(const Quat& q) const { return Quat(re - q.re, imx - q.imx, imy - q.imy, imz - q.imz); }
    SPR_CUDA_HOST_DEVICE_INLINE Quat& operator-=(Quat q) { re -= q.re; imx -= q.imx; imy -= q.imy; imz -= q.imz; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator-() const { return Quat(-re, -imx, -imy, -imz); }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator*(T scalar) const { return Quat(re * scalar, imx * scalar, imy * scalar, imz * scalar); }
    SPR_CUDA_HOST_DEVICE_INLINE Quat& operator*=(T scalar) { re *= scalar; imx *= scalar; imy *= scalar; imz *= scalar; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator/(T scalar) const { return Quat(re / scalar, imx / scalar, imy / scalar, imz / scalar); }
    SPR_CUDA_HOST_DEVICE_INLINE Quat& operator/=(T scalar) { re /= scalar; imx /= scalar; imy /= scalar; imz /= scalar; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator*(Quat q) const {
        return Quat(
            re * q.re - imx * q.imx - imy * q.imy - imz * q.imz,
            re * q.imx + imx * q.re + imy * q.imz - imz * q.imy,
            re * q.imy - imx * q.imz + imy * q.re + imz * q.imx,
            re * q.imz + imx * q.imy - imy * q.imx + imz * q.re
        );
    }

    SPR_CUDA_HOST_DEVICE_INLINE Quat& operator*=(Quat q) { *this = (*this) * q; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Quat conj() const { return Quat(re, -imx, -imy, -imz); }

    SPR_CUDA_HOST_DEVICE_INLINE Quat inv() const {
        const T n2 = sqNorm(*this);
        if constexpr (config::MATH_SAFETY) {
            const bool valid = (n2 > sprEpsilon<T>()) && sprIsFinite(n2);
            SPR_ASSERT(valid && "Quat::inv undefined for non-finite or near-zero norm");
            if (!valid) { return Quat::one(); }
        }
        return conj() / n2;
    }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator/(Quat q) const { return (*this) * q.inv(); }
    SPR_CUDA_HOST_DEVICE_INLINE Quat& operator/=(Quat q) { *this = (*this) / q; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Quat operator*(Vec<T, 3> v) const {
        return Quat(
            -(imx * v.x + imy * v.y + imz * v.z),
            (re * v.x - imz * v.y + imy * v.z),
            (imz * v.x + re * v.y - imx * v.z),
            (-imy * v.x + imx * v.y + re * v.z)
        );
    }

    SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 3> imVec() const { return Vec<T, 3>(imx, imy, imz); }

    SPR_CUDA_HOST_DEVICE_INLINE static Quat zero() { return Quat(T(0), T(0), T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Quat one() { return Quat(T(1), T(0), T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Quat i() { return Quat(T(0), T(1), T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Quat j() { return Quat(T(0), T(0), T(1), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Quat k() { return Quat(T(0), T(0), T(0), T(1)); }
};

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> operator*(T scalar, const Quat<T>& q) { return q * scalar; }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> operator*(Vec<T, 3> v, const Quat<T>& q) {
    return Quat<T>(
        -(v.x * q.imx + v.y * q.imy + v.z * q.imz),
        (v.x * q.re + v.y * q.imz - v.z * q.imy),
        (v.y * q.re - v.x * q.imz + v.z * q.imx),
        (v.z * q.re + v.x * q.imy - v.y * q.imx)
    );
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T sqNorm(Quat<T> q) {
    return q.re * q.re + q.imx * q.imx + q.imy * q.imy + q.imz * q.imz;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T length(Quat<T> q) { return sprSqrt<T>(sqNorm(q)); }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T dot(Quat<T> a, Quat<T> b) {
    return a.re * b.re + a.imx * b.imx + a.imy * b.imy + a.imz * b.imz;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE bool isUnit(Quat<T> q, T eps = T(8) * sprEpsilon<T>()) {
    return sprAbs(sqNorm(q) - T(1)) <= eps;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE void normalize(Quat<T>& q) {
    const T n2 = sqNorm(q);
    if constexpr (config::MATH_SAFETY) {
        if (!(n2 > T(0)) || !sprIsFinite(n2)) {
            q = Quat<T>::one();
            return;
        }
    }
    q *= sprRsqrt<T>(n2);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> normal(Quat<T> q) {
    normalize(q);
    return q;
}

/**
 * @brief General quaternion exponential.
 *
 * Computes
 *
 * ```text
 * exp(a + v) = exp(a) * (cos(|v|) + v/|v| * sin(|v|))
 * ```
 *
 * with a small-imaginary branch to avoid division by a tiny `|v|`.
 *
 * This is the algebraic quaternion exponential. For the SO(3) exponential map,
 * use `rot3Exp(w)`, which applies the required half-angle relation:
 * `rot3Exp(w)` is backed by `expImagQuat(0.5 * w)`.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> exp(Quat<T> q) {
    const Vec<T, 3> v = q.imVec();
    const T rho2 = sqNorm(v);
    const T ew = (q.re == T(0)) ? T(1) : sprExp<T>(q.re);

    if (rho2 <= T(1e-12)) {
        const T c = T(1) - T(0.5) * rho2;
        const T s = T(1) - (T(1) / T(6)) * rho2;
        return ew * Quat<T>(c, v.x * s, v.y * s, v.z * s);
    }

    const T rho = sprSqrt<T>(rho2);
    T sinRho = T(0);
    T cosRho = T(1);
    sprSincos<T>(rho, &sinRho, &cosRho);
    const T s = sinRho / rho;

    return ew * Quat<T>(cosRho, v.x * s, v.y * s, v.z * s);
}

/**
 * @brief Polynomial small-angle approximation for `exp(q)`.
 *
 * This variant uses fixed Taylor polynomials for `cos(|imag(q)|)` and
 * `sin(|imag(q)|)/|imag(q)|`. It is intended for callers who know the
 * imaginary part is small and have chosen the acceptable error/speed tradeoff.
 *
 * Prefer plain `exp(q)` when the input scale is not known.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> expSmall(Quat<T> q) {
    const Vec<T, 3> v = q.imVec();
    const T rho2 = sqNorm(v);
    const T rho4 = rho2 * rho2;
    const T rho6 = rho4 * rho2;
    const T c = T(1) - T(0.5) * rho2 + (T(1) / T(24)) * rho4 - (T(1) / T(720)) * rho6;
    const T s = T(1) - (T(1) / T(6)) * rho2 + (T(1) / T(120)) * rho4 - (T(1) / T(5040)) * rho6;
    const T ew = (q.re == T(0)) ? T(1) : sprExp<T>(q.re);
    return ew * Quat<T>(c, v.x * s, v.y * s, v.z * s);
}

/**
 * @brief Short suffix alias for `expSmall(q)`.
 *
 * The `_s` suffix means "small-input specialization". It exists to keep math
 * formulas compact while still signaling that the caller has selected a
 * specialized variant.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> exp_s(Quat<T> q) {
    return expSmall(q);
}

/**
 * @brief Exponential of a pure imaginary quaternion `exp(0 + v)`.
 *
 * This avoids constructing a temporary quaternion and uses the small-input
 * polynomial path when `|v|` is small enough.
 *
 * @note This is not the SO(3) exponential map. `rot3Exp(w)` uses
 * `expImagQuat(0.5 * w)` because unit quaternions store half the physical
 * rotation angle.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> expImagQuat(Vec<T, 3> v) {
    const T rho2 = sqNorm(v);
    if (rho2 <= T(0.0025)) {
        return expSmall(Quat<T>(v));
    }

    const T rho = sprSqrt<T>(rho2);
    T sinRho = T(0);
    T cosRho = T(1);
    sprSincos<T>(rho, &sinRho, &cosRho);
    const T s = sinRho / rho;
    return Quat<T>(cosRho, v.x * s, v.y * s, v.z * s);
}

/**
 * @brief Short suffix alias for `expImagQuat(v)`.
 *
 * The `_t` suffix means "tangent-vector input": `v` is interpreted as the
 * imaginary part of a quaternion. For SO(3) tangent vectors, prefer
 * `rot3Exp(w)` so the half-angle convention is applied for you.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> exp_t(Vec<T, 3> v) {
    return expImagQuat(v);
}

/**
 * @brief General quaternion logarithm.
 *
 * Returns a quaternion whose real part is `0.5 * log(sqNorm(q))` and whose
 * imaginary part points along `imag(q)`.
 *
 * Degenerate near-zero quaternions return zero. Unit quaternions that are being
 * used as rotations should normally be handled through `rot3Log(R)`; if a raw
 * unit quaternion is already available, `log_u(q)` avoids the norm/log work.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> log(Quat<T> q) {
    const Vec<T, 3> v = q.imVec();
    const T rho2 = sqNorm(v);
    const T q2 = sqNorm(q);
    const T re = q.re;

    if (q2 <= sprEpsilon<T>() * sprEpsilon<T>()) {
        return Quat<T>::zero();
    }

    const T real = T(0.5) * sprLog<T>(q2);

    if (rho2 < T(1e-12)) {
        if (sprAbs(re) > T(1e-6)) {
            const T k = T(1) / re;
            return Quat<T>(real, v.x * k, v.y * k, v.z * k);
        }
        return Quat<T>(real, T(0), T(0), T(0));
    }

    const T rho = sprSqrt<T>(rho2);
    const T theta = sprAtan2<T>(rho, re);
    const T k = theta / rho;

    return Quat<T>(real, v.x * k, v.y * k, v.z * k);
}

/**
 * @brief Specialized logarithm for unit quaternions.
 *
 * Returns only `imag(log(q))`, skipping the real component and the
 * `log(sqNorm(q))` computation because a unit quaternion has zero logarithmic
 * real part. In safety mode the input is normalized first to repair small
 * drift; with math safety disabled, the caller is responsible for unit length.
 *
 * For rotations, `rot3Log(R)` first chooses the shortest quaternion
 * representative and returns `2 * log_u(q)`.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 3> log_u(Quat<T> q) {
    if constexpr (config::MATH_SAFETY) {
        normalize(q);
    }

    const Vec<T, 3> v = q.imVec();
    const T rho2 = sqNorm(v);
    if (rho2 < T(1e-12)) {
        if (sprAbs(q.re) > T(1e-6)) {
            const T k = T(1) / q.re;
            return Vec<T, 3>(v.x * k, v.y * k, v.z * k);
        }
        return Vec<T, 3>::zero();
    }

    const T rho = sprSqrt<T>(rho2);
    const T theta = sprAtan2<T>(rho, q.re);
    return (theta / rho) * v;
}

/**
 * @brief Rotate a vector by a known-unit quaternion without checks.
 *
 * This is a low-level helper for `Rot<T, 3>`. It assumes `q` is unit length and
 * uses the direct vector formula instead of forming `q * v * conj(q)`.
 *
 * Prefer `rot3 * vec3` in public code.
 */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 3> rotateByUnitQuatUnchecked(Vec<T, 3> v, Quat<T> q) {
    const Vec<T, 3> u(q.imx, q.imy, q.imz);
    const Vec<T, 3> t = T(2) * cross(u, v);
    return v + q.re * t + cross(u, t);
}

using quat = Quat<f32>;
using dquat = Quat<f64>;
