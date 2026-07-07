#pragma once

/**
 * @file complex.cuh
 * @brief Algebraic complex numbers used by scalar math and 2D rotations.
 *
 * `Complex<T>` is the general complex-number type. It may have any norm. Use
 * `Rot<T, 2>` when a value semantically represents an element of SO(2).
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "scalar_math.cuh"
#include "vec.cuh"

/** Complex algebra value `re + im*i`. */
template<class T>
struct alignas(VecAlign<2, T>::value) Complex {
    T re;
    T im;

    SPR_CUDA_HOST_DEVICE_INLINE Complex() : re(T(0)), im(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Complex(T a) : re(a), im(T(0)) {}
    SPR_CUDA_HOST_DEVICE_INLINE Complex(T re_, T im_) : re(re_), im(im_) {}
    SPR_CUDA_HOST_DEVICE_INLINE explicit Complex(const Vec<T, 2>& z) : re(z.x), im(z.y) {}

    SPR_CUDA_HOST_DEVICE_INLINE explicit operator Vec<T, 2>() const { return Vec<T, 2>(re, im); }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator+(const Complex& z) const { return Complex(re + z.re, im + z.im); }
    SPR_CUDA_HOST_DEVICE_INLINE Complex& operator+=(Complex z) { re += z.re; im += z.im; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator-(const Complex& z) const { return Complex(re - z.re, im - z.im); }
    SPR_CUDA_HOST_DEVICE_INLINE Complex& operator-=(Complex z) { re -= z.re; im -= z.im; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator-() const { return Complex(-re, -im); }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator*(T scalar) const { return Complex(re * scalar, im * scalar); }
    SPR_CUDA_HOST_DEVICE_INLINE Complex& operator*=(T scalar) { re *= scalar; im *= scalar; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator/(T scalar) const { return Complex(re / scalar, im / scalar); }
    SPR_CUDA_HOST_DEVICE_INLINE Complex& operator/=(T scalar) { re /= scalar; im /= scalar; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator*(Complex z) const {
        return Complex(re * z.re - im * z.im, re * z.im + im * z.re);
    }

    SPR_CUDA_HOST_DEVICE_INLINE Complex& operator*=(Complex z) { *this = (*this) * z; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Complex conj() const { return Complex(re, -im); }

    SPR_CUDA_HOST_DEVICE_INLINE Complex inv() const {
        const T n2 = sqNorm(*this);
        if constexpr (config::MATH_SAFETY) {
            const bool valid = (n2 > sprEpsilon<T>()) && sprIsFinite(n2);
            SPR_ASSERT(valid && "Complex::inv undefined for non-finite or near-zero norm");
            if (!valid) { return Complex::one(); }
        }
        return conj() / n2;
    }

    SPR_CUDA_HOST_DEVICE_INLINE Complex operator/(Complex z) const { return (*this) * z.inv(); }
    SPR_CUDA_HOST_DEVICE_INLINE Complex& operator/=(Complex z) { *this = (*this) / z; return *this; }

    SPR_CUDA_HOST_DEVICE_INLINE Complex imRe() const { return Complex(im, re); }

    SPR_CUDA_HOST_DEVICE_INLINE static Complex zero() { return Complex(T(0), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Complex one() { return Complex(T(1), T(0)); }
    SPR_CUDA_HOST_DEVICE_INLINE static Complex i() { return Complex(T(0), T(1)); }
};

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> operator*(T scalar, const Complex<T>& z) { return z * scalar; }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T sqNorm(Complex<T> z) { return z.re * z.re + z.im * z.im; }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T length(Complex<T> z) { return sprSqrt<T>(sqNorm(z)); }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T dot(Complex<T> a, Complex<T> b) { return a.re * b.re + a.im * b.im; }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T cross(Complex<T> a, Complex<T> b) { return a.re * b.im - a.im * b.re; }

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE bool isUnit(Complex<T> z, T eps = T(8) * sprEpsilon<T>()) {
    return sprAbs(sqNorm(z) - T(1)) <= eps;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE void normalize(Complex<T>& z) {
    const T n2 = sqNorm(z);
    if constexpr (config::MATH_SAFETY) {
        if (!(n2 > T(0)) || !sprIsFinite(n2)) {
            z = Complex<T>::one();
            return;
        }
    }
    z *= sprRsqrt<T>(n2);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> normal(Complex<T> z) {
    normalize(z);
    return z;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> exp(Complex<T> z) {
    const T ez = sprExp<T>(z.re);
    T s = T(0);
    T c = T(1);
    sprSincos<T>(z.im, &s, &c);
    return Complex<T>(ez * c, ez * s);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> log(Complex<T> z) {
    return Complex<T>(sprLog<T>(length(z)), sprAtan2<T>(z.im, z.re));
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> pow(Complex<T> z, Complex<T> w) {
    return exp(w * log(z));
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> cos(Complex<T> z) {
    return (exp(Complex<T>::i() * z) + exp(-Complex<T>::i() * z)) * T(0.5);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Complex<T> sin(Complex<T> z) {
    return (exp(Complex<T>::i() * z) - exp(-Complex<T>::i() * z)) * Complex<T>(T(0), T(-0.5));
}

using complex = Complex<f32>;
using dcomplex = Complex<f64>;
