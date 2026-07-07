#pragma once

/**
 * @file mat.cuh
 * @brief Fixed-size row-major matrices and small matrix operations.
 *
 * `Mat<T, R, C>` supports both square and rectangular matrices. It is designed
 * for small dimensions used directly inside formulas and CUDA kernels, not as a
 * replacement for large dense linear algebra libraries.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "scalar_math.cuh"
#include "vec.cuh"

#include <type_traits>

/**
 * @brief Fixed-size row-major matrix.
 *
 * Square matrices are spelled `Mat<T, N>`; rectangular matrices are
 * `Mat<T, R, C>`. The flat `data` member is row-major, and `operator[](r)[c]`
 * is preserved through a row proxy for formula readability.
 */
template<class T, u32 R, u32 C = R>
struct Mat {
    static_assert(R > 0, "Mat<T, R, C> requires R > 0");
    static_assert(C > 0, "Mat<T, R, C> requires C > 0");
    static_assert(std::is_arithmetic_v<T>, "Mat<T, R, C> requires arithmetic scalar T");

    T data[R * C];

    struct RowRef {
        T* p;
        SPR_CUDA_HOST_DEVICE_INLINE T& operator[](u32 c) { return p[c]; }
    };

    struct ConstRowRef {
        const T* p;
        SPR_CUDA_HOST_DEVICE_INLINE const T& operator[](u32 c) const { return p[c]; }
    };

    SPR_CUDA_HOST_DEVICE_INLINE Mat() {
        SPR_UNROLL
        for (u32 i = 0; i < R * C; ++i) { data[i] = T(0); }
    }

    SPR_CUDA_HOST_DEVICE_INLINE explicit Mat(T a) {
        SPR_UNROLL
        for (u32 i = 0; i < R * C; ++i) { data[i] = a; }
    }

    SPR_CUDA_HOST_DEVICE_INLINE explicit Mat(const T* array) {
        SPR_UNROLL
        for (u32 i = 0; i < R * C; ++i) { data[i] = array[i]; }
    }

    SPR_CUDA_HOST_DEVICE_INLINE T& operator()(u32 r, u32 c) { return data[r * C + c]; }
    SPR_CUDA_HOST_DEVICE_INLINE const T& operator()(u32 r, u32 c) const { return data[r * C + c]; }

    SPR_CUDA_HOST_DEVICE_INLINE RowRef operator[](u32 r) { return RowRef{ data + r * C }; }
    SPR_CUDA_HOST_DEVICE_INLINE ConstRowRef operator[](u32 r) const { return ConstRowRef{ data + r * C }; }

    SPR_CUDA_HOST_DEVICE_INLINE static Mat zero() { return Mat(T(0)); }

    SPR_CUDA_HOST_DEVICE_INLINE static Mat eye() {
        static_assert(R == C, "Mat::eye requires a square matrix");
        Mat out;
        SPR_UNROLL
        for (u32 i = 0; i < R; ++i) { out[i][i] = T(1); }
        return out;
    }
};

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator+(const Mat<T, R, C>& A, const Mat<T, R, C>& B) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { out.data[i] = A.data[i] + B.data[i]; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C>& operator+=(Mat<T, R, C>& A, const Mat<T, R, C>& B) {
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { A.data[i] += B.data[i]; }
    return A;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator-(const Mat<T, R, C>& A, const Mat<T, R, C>& B) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { out.data[i] = A.data[i] - B.data[i]; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C>& operator-=(Mat<T, R, C>& A, const Mat<T, R, C>& B) {
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { A.data[i] -= B.data[i]; }
    return A;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator-(const Mat<T, R, C>& A) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { out.data[i] = -A.data[i]; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator*(const Mat<T, R, C>& A, T s) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { out.data[i] = A.data[i] * s; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator*(T s, const Mat<T, R, C>& A) { return A * s; }

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C>& operator*=(Mat<T, R, C>& A, T s) {
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { A.data[i] *= s; }
    return A;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator/(const Mat<T, R, C>& A, T s) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 i = 0; i < R * C; ++i) { out.data[i] = A.data[i] / s; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, R> operator*(const Mat<T, R, C>& A, const Vec<T, C>& x) {
    Vec<T, R> out;
    SPR_UNROLL
    for (u32 r = 0; r < R; ++r) {
        T sum = T(0);
        SPR_UNROLL
        for (u32 c = 0; c < C; ++c) { sum += A[r][c] * x[c]; }
        out[r] = sum;
    }
    return out;
}

template<class T, u32 R, u32 K, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> operator*(const Mat<T, R, K>& A, const Mat<T, K, C>& B) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 r = 0; r < R; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < C; ++c) {
            T sum = T(0);
            SPR_UNROLL
            for (u32 k = 0; k < K; ++k) { sum += A[r][k] * B[k][c]; }
            out[r][c] = sum;
        }
    }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, C, R> transpose(const Mat<T, R, C>& A) {
    Mat<T, C, R> out;
    SPR_UNROLL
    for (u32 r = 0; r < R; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < C; ++c) { out[c][r] = A[r][c]; }
    }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, C> row(const Mat<T, R, C>& A, u32 r) {
    Vec<T, C> out;
    SPR_UNROLL
    for (u32 c = 0; c < C; ++c) { out[c] = A[r][c]; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Vec<T, R> col(const Mat<T, R, C>& A, u32 c) {
    Vec<T, R> out;
    SPR_UNROLL
    for (u32 r = 0; r < R; ++r) { out[r] = A[r][c]; }
    return out;
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE void setRow(Mat<T, R, C>& A, u32 r, const Vec<T, C>& v) {
    SPR_UNROLL
    for (u32 c = 0; c < C; ++c) { A[r][c] = v[c]; }
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE void setCol(Mat<T, R, C>& A, u32 c, const Vec<T, R>& v) {
    SPR_UNROLL
    for (u32 r = 0; r < R; ++r) { A[r][c] = v[r]; }
}

template<class T, u32 R, u32 C>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, R, C> outer(const Vec<T, R>& a, const Vec<T, C>& b) {
    Mat<T, R, C> out;
    SPR_UNROLL
    for (u32 r = 0; r < R; ++r) {
        SPR_UNROLL
        for (u32 c = 0; c < C; ++c) { out[r][c] = a[r] * b[c]; }
    }
    return out;
}

template<class T, u32 N>
SPR_CUDA_HOST_DEVICE_INLINE T trace(const Mat<T, N, N>& A) {
    T sum = T(0);
    SPR_UNROLL
    for (u32 i = 0; i < N; ++i) { sum += A[i][i]; }
    return sum;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T det(const Mat<T, 2, 2>& A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, 2, 2> inverse(const Mat<T, 2, 2>& A) {
    const T d = det(A);
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(sprAbs(d) > sprEpsilon<T>() && "inverse(mat2) singular or near-singular");
    }

    Mat<T, 2, 2> out;
    const T invD = T(1) / d;
    out[0][0] =  A[1][1] * invD;
    out[0][1] = -A[0][1] * invD;
    out[1][0] = -A[1][0] * invD;
    out[1][1] =  A[0][0] * invD;
    return out;
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE T det(const Mat<T, 3, 3>& A) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

template<class T>
SPR_CUDA_HOST_DEVICE_INLINE Mat<T, 3, 3> inverse(const Mat<T, 3, 3>& A) {
    const T d = det(A);
    if constexpr (config::MATH_SAFETY) {
        SPR_ASSERT(sprAbs(d) > sprEpsilon<T>() && "inverse(mat3) singular or near-singular");
    }

    const T invD = T(1) / d;
    Mat<T, 3, 3> out;

    out[0][0] =  (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * invD;
    out[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * invD;
    out[0][2] =  (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * invD;

    out[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * invD;
    out[1][1] =  (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * invD;
    out[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) * invD;

    out[2][0] =  (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * invD;
    out[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) * invD;
    out[2][2] =  (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * invD;

    return out;
}

template<u32 R, u32 C = R>
using mat = Mat<f32, R, C>;

using mat2 = Mat<f32, 2>;
using mat3 = Mat<f32, 3>;
using mat4 = Mat<f32, 4>;

using mat2x3 = Mat<f32, 2, 3>;
using mat3x2 = Mat<f32, 3, 2>;
using mat3x4 = Mat<f32, 3, 4>;
using mat4x3 = Mat<f32, 4, 3>;

using dmat2 = Mat<f64, 2>;
using dmat3 = Mat<f64, 3>;
using dmat4 = Mat<f64, 4>;
