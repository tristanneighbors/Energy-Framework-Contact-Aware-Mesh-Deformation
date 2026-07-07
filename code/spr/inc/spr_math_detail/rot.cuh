#pragma once

/**
 * @file rot.cuh
 * @brief Rotation group wrappers for SO(N), with compact SO(2), SO(3), and SO(4) storage.
 *
 * `Rot<T, N>` is the semantic rotation type. Generic dimensions are
 * matrix-backed; SO(2) uses a unit complex value, SO(3) uses a unit quaternion,
 * and SO(4) uses left/right unit quaternions. Constructors with `Unchecked` in
 * the name trust their invariants; checked constructors assert in math safety
 * mode, and normalizing constructors repair inputs.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "scalar_math.cuh"
#include "vec.cuh"
#include "mat.cuh"
#include "complex.cuh"
#include "quat.cuh"

/** Matrix-backed rotation in SO(N). */
template<class T, u32 N>
struct Rot {
    Mat<T, N, N> m;

    SPR_CUDA_HOST_DEVICE_INLINE Rot() : m(Mat<T, N, N>::eye()) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Rot(const Mat<T, N, N>& m_) : m(m_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Rot identity() { return Rot(); }
};

/** SO(2) rotation stored as a unit complex number. */
template<class T>
struct Rot<T, 2> {
    Complex<T> z; // invariant: unit complex

    SPR_CUDA_HOST_DEVICE_INLINE Rot() : z(Complex<T>::one()) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Rot(Complex<T> z_) : z(z_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Rot identity() { return Rot(); }
};

/** SO(3) rotation stored as a unit quaternion. */
template<class T>
struct Rot<T, 3> {
    Quat<T> q; // invariant: unit quaternion

    SPR_CUDA_HOST_DEVICE_INLINE Rot() : q(Quat<T>::one()) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Rot(Quat<T> q_) : q(q_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Rot identity() { return Rot(); }
};

/** SO(4) rotation stored as a pair of unit quaternions. */
template<class T>
struct Rot<T, 4> {
    Quat<T> l; // invariant: unit quaternion
    Quat<T> r; // invariant: unit quaternion

    SPR_CUDA_HOST_DEVICE_INLINE Rot() : l(Quat<T>::one()), r(Quat<T>::one()) {}
    SPR_CUDA_HOST_DEVICE_INLINE Rot(Quat<T> l_, Quat<T> r_) : l(l_), r(r_) {}

    SPR_CUDA_HOST_DEVICE_INLINE static Rot identity() { return Rot(); }
};

// ----------------------------- generic matrix-backed Rot<T,N>

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, N> rotFromMatrixUnchecked(const Mat<T, N, N>& m) {
    return Rot<T, N>(m);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, N> rotFromOrthonormalMatrix(const Mat<T, N, N>& m) {
    return Rot<T, N>(m);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, N> operator*(Rot<T, N> R, Vec<T, N> x) {
    return R.m * x;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, N> operator*(Rot<T, N> A, Rot<T, N> B) {
    return Rot<T, N>(A.m * B.m);
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, N> inverse(Rot<T, N> R) {
    return Rot<T, N>(transpose(R.m));
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, N, N> matrix(Rot<T, N> R) {
    return R.m;
}

// ----------------------------- Rot<T,2>

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 2> rot2FromUnitComplexUnchecked(Complex<T> z) {
    return Rot<T, 2>(z);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 2> rot2FromUnitComplex(Complex<T> z) {
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(isUnit(z) && "rot2FromUnitComplex requires unit complex input");
    }
    return Rot<T, 2>(z);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 2> rot2FromComplex(Complex<T> z) {
    normalize(z);
    return Rot<T, 2>(z);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 2> rot2FromAngle(T angle) {
    T s = T(0);
    T c = T(1);
    sprSincos<T>(angle, &s, &c);
    return Rot<T, 2>(Complex<T>(c, s));
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 2> operator*(Rot<T, 2> R, Vec<T, 2> v) {
    return Vec<T, 2>(
        R.z.re * v.x - R.z.im * v.y,
        R.z.im * v.x + R.z.re * v.y
    );
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 2> operator*(Rot<T, 2> A, Rot<T, 2> B) {
    Complex<T> z = A.z * B.z;
    if constexpr (config::MATH_SAFETY) { normalize(z); }
    return Rot<T, 2>(z);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 2> inverse(Rot<T, 2> R) {
    return Rot<T, 2>(R.z.conj());
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE void renormalize(Rot<T, 2>& R) { normalize(R.z); }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, 2, 2> matrix(Rot<T, 2> R) {
    Mat<T, 2, 2> out;
    out[0][0] = R.z.re;
    out[0][1] = -R.z.im;
    out[1][0] = R.z.im;
    out[1][1] = R.z.re;
    return out;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T angle(Rot<T, 2> R) {
    return sprAtan2<T>(R.z.im, R.z.re);
}

// ----------------------------- Rot<T,3>

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> rot3FromUnitQuatUnchecked(Quat<T> q) {
    return Rot<T, 3>(q);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> rot3FromUnitQuat(Quat<T> q) {
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(isUnit(q) && "rot3FromUnitQuat requires unit quaternion input");
    }
    return Rot<T, 3>(q);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> rot3FromQuat(Quat<T> q) {
    normalize(q);
    return Rot<T, 3>(q);
}

template<class T, class Angle>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> rot3FromUnitAxisAngle(Vec<T, 3> unitAxis, Angle angle) {
    if constexpr (config::MATH_SAFETY) {
        const bool valid = isUnit(unitAxis);
        SPR_ASSERT(valid && "rot3FromUnitAxisAngle requires unit axis");
        if (!valid) { return Rot<T, 3>::identity(); }
    }

    const T halfAngle = T(0.5) * T(angle);
    T s = T(0);
    T c = T(1);
    sprSincos<T>(halfAngle, &s, &c);
    return Rot<T, 3>(Quat<T>(c, unitAxis.x * s, unitAxis.y * s, unitAxis.z * s));
}

template<class T, class Angle>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> rot3FromAxisAngle(Vec<T, 3> axis, Angle angle) {
    if constexpr (config::MATH_SAFETY) {
        const T n2 = sqNorm(axis);
        if (!(n2 > sprEpsilon<T>()) || !sprIsFinite(n2)) {
            return Rot<T, 3>::identity();
        }
    }
    normalize(axis);
    return rot3FromUnitAxisAngle(axis, angle);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Quat<T> quatFromRot3(Rot<T, 3> R) { return R.q; }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 3> operator*(Rot<T, 3> R, Vec<T, 3> v) {
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(isUnit(R.q) && "rot3 drifted from unit quaternion invariant");
    }
    return rotateByUnitQuatUnchecked(v, R.q);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> operator*(Rot<T, 3> A, Rot<T, 3> B) {
    Quat<T> q = A.q * B.q;
    if constexpr (config::MATH_SAFETY) { normalize(q); }
    return Rot<T, 3>(q);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> inverse(Rot<T, 3> R) {
    return Rot<T, 3>(R.q.conj());
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE void renormalize(Rot<T, 3>& R) { normalize(R.q); }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, 3, 3> matrix(Rot<T, 3> R) {
    Quat<T> q = R.q;
    if constexpr (config::MATH_SAFETY) { normalize(q); }

    const T x = q.imx;
    const T y = q.imy;
    const T z = q.imz;
    const T w = q.re;

    Mat<T, 3, 3> out;
    out[0][0] = T(1) - T(2) * (y * y + z * z);
    out[0][1] = T(2) * (x * y - z * w);
    out[0][2] = T(2) * (x * z + y * w);

    out[1][0] = T(2) * (x * y + z * w);
    out[1][1] = T(1) - T(2) * (x * x + z * z);
    out[1][2] = T(2) * (y * z - x * w);

    out[2][0] = T(2) * (x * z - y * w);
    out[2][1] = T(2) * (y * z + x * w);
    out[2][2] = T(1) - T(2) * (x * x + y * y);

    return out;
}

/** SO(3) exponential map from an axis-angle vector with `|w|` in radians. */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 3> rot3Exp(Vec<T, 3> w) {
    return Rot<T, 3>(expImagQuat(T(0.5) * w));
}

/** Shortest SO(3) logarithm; `q` and `-q` are treated as the same rotation. */
template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 3> rot3Log(Rot<T, 3> R) {
    Quat<T> q = R.q;
    if (q.re < T(0)) { q = -q; }
    return T(2) * log_u(q);
}

// ----------------------------- Rot<T,4>

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 4> rot4FromUnitLeftRightUnchecked(Quat<T> l, Quat<T> r) {
    return Rot<T, 4>(l, r);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 4> rot4FromUnitLeftRight(Quat<T> l, Quat<T> r) {
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(isUnit(l) && "rot4FromUnitLeftRight requires unit left quaternion");
        SPR_ASSERT(isUnit(r) && "rot4FromUnitLeftRight requires unit right quaternion");
    }
    return Rot<T, 4>(l, r);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 4> rot4FromLeftRight(Quat<T> l, Quat<T> r) {
    normalize(l);
    normalize(r);
    return Rot<T, 4>(l, r);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, 4> operator*(Rot<T, 4> R, Vec<T, 4> v) {
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(isUnit(R.l) && "rot4 left quaternion drifted from unit invariant");
        SPR_ASSERT(isUnit(R.r) && "rot4 right quaternion drifted from unit invariant");
    }

    // Convention: vec4(x,y,z,w) maps to quaternion w + x*i + y*j + z*k.
    Quat<T> x(v);
    Quat<T> y = R.l * x * R.r.conj();
    return Vec<T, 4>(y.imx, y.imy, y.imz, y.re);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 4> operator*(Rot<T, 4> A, Rot<T, 4> B) {
    // Apply B first, then A.
    Quat<T> l = A.l * B.l;
    Quat<T> r = A.r * B.r;
    if constexpr (config::MATH_SAFETY) {
        normalize(l);
        normalize(r);
    }
    return Rot<T, 4>(l, r);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Rot<T, 4> inverse(Rot<T, 4> R) {
    return Rot<T, 4>(R.l.conj(), R.r.conj());
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE void renormalize(Rot<T, 4>& R) {
    normalize(R.l);
    normalize(R.r);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, 4, 4> matrix(Rot<T, 4> R) {
    Mat<T, 4, 4> out;
    SPR_UNROLL
    for (u32 c = 0; c < 4; ++c) {
        Vec<T, 4> e(T(0));
        e[c] = T(1);
        Vec<T, 4> image = R * e;
        SPR_UNROLL
        for (u32 r = 0; r < 4; ++r) {
            out[r][c] = image[r];
        }
    }
    return out;
}

template<u32 N>
using rot = Rot<f32, N>;

using rot2 = Rot<f32, 2>;
using rot3 = Rot<f32, 3>;
using rot4 = Rot<f32, 4>;
using rot5 = Rot<f32, 5>;

using drot2 = Rot<f64, 2>;
using drot3 = Rot<f64, 3>;
using drot4 = Rot<f64, 4>;
