#pragma once

/**
 * @file containers.hpp
 * @brief Lightweight owning and non-owning contiguous containers.
 *
 * The `containers` namespace provides fixed arrays, spans, movable heap blocks,
 * and row-major N-D views. Bounds checks are controlled by `config::DEBUG`.
 */

#include "spr_global_include.h"
#include "config.hpp"

#include <algorithm> // std::copy
#include <cassert>   // assert
#include <type_traits> // std::remove_cv_t
#include <utility>   // std::exchange

namespace containers {

template<u64... Dims, class... Ix>
constexpr u64 containerLinearIndex(Ix... ix) {
    static_assert(sizeof...(Dims) > 0);
    static_assert(sizeof...(Ix) == sizeof...(Dims));

    constexpr u64 rank = sizeof...(Dims);
    u64 idx[rank]  = { u64(ix)... };
    u64 dims[rank] = { u64(Dims)... };

    if constexpr (config::DEBUG) {
        for (u64 k = 0; k < rank; ++k) { assert(idx[k] < dims[k]); }
    }

    u64 lin = 0;
    u64 stride = 1;
    for (u64 k = rank; k-- > 0;) {
        lin += idx[k] * stride;
        stride *= dims[k];
    }
    return lin;
}

template<class T>
using ContainerElem = std::remove_cv_t<T>;

template<typename T> struct Block; // forward declaration for use in Span

/** Non-owning contiguous view. */
template<typename T>
struct Span {
    T* data = nullptr;
    u64 size = 0;

    constexpr Span() = default;
    constexpr Span(T* p, u64 n) : data(p), size(n) {}

    constexpr T& operator[](u64 i) {
        if constexpr (config::DEBUG) { assert(i < size); }
        return data[i];
    }
    constexpr const T& operator[](u64 i) const {
        if constexpr (config::DEBUG) { assert(i < size); }
        return data[i];
    }
    
    constexpr T* begin() { return data; }
    constexpr T* end() { return data + size; }
    constexpr const T* begin() const { return data; }
    constexpr const T* end() const { return data + size; }
    constexpr const T* cbegin() const { return data; }
    constexpr const T* cend() const { return data + size; }
};


/** Fixed-size owning array with public contiguous storage. */
template<typename T, u64 N>
struct Array {
    static constexpr u64 size = N;
    T data[N];

    constexpr Array() = default;

    template<class... U, class = std::enable_if_t<sizeof...(U) == N && (std::is_convertible_v<U, T> && ...)>>
    constexpr Array(U... values) : data{ static_cast<T>(values)... } {}

    constexpr T& operator[](u64 i) {
        if constexpr (config::DEBUG) { assert(i < N); }
        return data[i];
    }
    constexpr const T& operator[](u64 i) const {
        if constexpr (config::DEBUG) { assert(i < N); }
        return data[i];
    }

    constexpr T* begin() { return data; }
    constexpr T* end() { return data + N; }
    constexpr const T* begin() const { return data; }
    constexpr const T* end() const { return data + N; }
};


/** Owning heap buffer: non-copyable, movable. */
template<typename T>
struct Block {
    T* data = nullptr;
    u64 size = 0;

    explicit Block(u64 n) : size(n), data(new T[n]) {}
    ~Block() noexcept { delete[] data; }
    
    // non-copyable (use span/subspan for views) but movable
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    Block(Block&& other) noexcept : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)) {}
    Block& operator=(Block&& other) noexcept {
        if (this == &other) { return *this; }
        delete[] data;
        data = std::exchange(other.data, nullptr);
        size = std::exchange(other.size, 0);
        return *this;
    }
    
    T& operator[](u64 idx) {
        if constexpr (config::DEBUG) {
            assert(idx < size && "index out of bounds");
        }
        return data[idx];
    }
    const T& operator[](u64 idx) const {
        if constexpr (config::DEBUG) {
            assert(idx < size && "index out of bounds");
        }
        return data[idx];
    }
    
    T* begin() { return data; }
    T* end() { return data + size; }
    const T* begin() const { return data; }
    const T* end() const { return data + size; }
    const T* cbegin() const { return data; }
    const T* cend() const { return data + size; }
};

// ----- slicing helpers (free functions) -----
// These intentionally use short lower-case names.

template<typename T>
[[nodiscard]] constexpr Span<T> span(Span<T> s, u64 st, u64 len) {
    if constexpr (config::DEBUG) { assert(st <= s.size && len <= s.size && st <= s.size - len); }
    return Span<T>(s.data + st, len);
}

template<typename T>
[[nodiscard]] constexpr Span<T> span(Span<T> s, u64 st) {
    return span(s, st, s.size - st);
}

template<typename T>
[[nodiscard]] constexpr Span<T> span(Span<T> s) {
    return s;
}

template<typename T>
[[nodiscard]] constexpr Span<T> subspan(Span<T> s, u64 st, u64 len) {
    return span(s, st, len);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<T> span(Array<T, N>& a, u64 st, u64 len) {
    if constexpr (config::DEBUG) { assert(st <= N && len <= N && st <= N - len); }
    return Span<T>(a.data + st, len);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<const T> span(const Array<T, N>& a, u64 st, u64 len) {
    if constexpr (config::DEBUG) { assert(st <= N && len <= N && st <= N - len); }
    return Span<const T>(a.data + st, len);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<T> span(Array<T, N>& a, u64 st) {
    return span(a, st, N - st);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<const T> span(const Array<T, N>& a, u64 st) {
    return span(a, st, N - st);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<T> span(Array<T, N>& a) {
    return span(a, 0, N);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<const T> span(const Array<T, N>& a) {
    return span(a, 0, N);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<T> subspan(Array<T, N>& a, u64 st, u64 len) {
    return span(a, st, len);
}

template<typename T, u64 N>
[[nodiscard]] constexpr Span<const T> subspan(const Array<T, N>& a, u64 st, u64 len) {
    return span(a, st, len);
}

template<typename T>
[[nodiscard]] inline Span<T> span(Block<T>& b, u64 st, u64 len) {
    if constexpr (config::DEBUG) { assert(st <= b.size && len <= b.size && st <= b.size - len); }
    return Span<T>(b.data + st, len);
}

template<typename T>
[[nodiscard]] inline Span<const T> span(const Block<T>& b, u64 st, u64 len) {
    if constexpr (config::DEBUG) { assert(st <= b.size && len <= b.size && st <= b.size - len); }
    return Span<const T>(b.data + st, len);
}

template<typename T>
[[nodiscard]] inline Span<T> span(Block<T>& b, u64 st) {
    return span(b, st, b.size - st);
}

template<typename T>
[[nodiscard]] inline Span<const T> span(const Block<T>& b, u64 st) {
    return span(b, st, b.size - st);
}

template<typename T>
[[nodiscard]] inline Span<T> span(Block<T>& b) {
    return span(b, 0, b.size);
}

template<typename T>
[[nodiscard]] inline Span<const T> span(const Block<T>& b) {
    return span(b, 0, b.size);
}

template<typename T>
[[nodiscard]] inline Span<T> subspan(Block<T>& b, u64 st, u64 len) {
    return span(b, st, len);
}

template<typename T>
[[nodiscard]] inline Span<const T> subspan(const Block<T>& b, u64 st, u64 len) {
    return span(b, st, len);
}

template<typename T, u64... Dims>
struct ArrayND;

/** Fixed-size row-major N-D array backed by a flat `Array`. */
template<typename T, u64... Dims>
struct ArrayND {
    static_assert(sizeof...(Dims) > 0);

    static constexpr u64 rank = sizeof...(Dims);
    static constexpr u64 size = (Dims * ...);

    Array<T, size> a{}; // row-major flat storage

    // Flat access
    constexpr Array<T, size>& flat() { return a; }
    constexpr const Array<T, size>& flat() const { return a; }

    template<typename... Ix>
    constexpr T& operator()(Ix... ix) {
        static_assert(sizeof...(Ix) == rank);
        const u64 lin = linearIndex(u64(ix)...);
        return a.data[lin];
    }

    template<typename... Ix>
    constexpr const T& operator()(Ix... ix) const {
        static_assert(sizeof...(Ix) == rank);
        const u64 lin = linearIndex(u64(ix)...);
        return a.data[lin];
    }

    template<typename... Ix>
    static constexpr u64 linearIndex(Ix... ix) {
        // Row-major: last dimension varies fastest
        return containerLinearIndex<Dims...>(ix...);
    }
};

/** Non-owning N-D view over contiguous row-major memory. */
template<class T, u64... Dims>
struct ArrayNDView {
    static_assert(sizeof...(Dims) > 0);

    static constexpr u64 rank = sizeof...(Dims);
    static constexpr u64 size = (Dims * ...);

    T* data = nullptr;

    constexpr ArrayNDView() = default;
    constexpr explicit ArrayNDView(T* p) : data(p) {}

    // Flattening (row-major) view
    [[nodiscard]] constexpr Span<T> flat() { return Span<T>(data, size); }
    [[nodiscard]] constexpr Span<const T> flat() const { return Span<const T>(data, size); }

    template<class... Ix>
    constexpr T& operator()(Ix... ix) {
        static_assert(sizeof...(Ix) == rank);
        return data[linearIndex(u64(ix)...)];
    }

    template<class... Ix>
    constexpr const T& operator()(Ix... ix) const {
        static_assert(sizeof...(Ix) == rank);
        return data[linearIndex(u64(ix)...)];
    }

    template<class... Ix>
    static constexpr u64 linearIndex(Ix... ix) {
        return containerLinearIndex<Dims...>(ix...);
    }
};

// helpers
template<typename T, u64... Dims>
constexpr ArrayNDView<T, Dims...> as_nd(Span<T> s) {
    if constexpr (config::DEBUG) { assert(s.size == (Dims * ...)); }
    return ArrayNDView<T, Dims...>(s.data);
}

template<typename T, u64... Dims>
constexpr ArrayNDView<const T, Dims...> as_nd(Span<const T> s) {
    if constexpr (config::DEBUG) { assert(s.size == (Dims * ...)); }
    return ArrayNDView<const T, Dims...>(s.data);
}

// ----- copy helpers -----

template<typename T>
Block<ContainerElem<T>> copy(Span<T> src, u64 st, u64 len, u64 padding = 0) {
    if constexpr (config::DEBUG) {
        assert(st <= src.size && len <= src.size && st <= src.size - len && "slice out of bounds");
    }
    Block<ContainerElem<T>> out(len + padding);
    std::copy(src.data + st, src.data + st + len, out.data);
    return out;
}

template<typename T>
Block<ContainerElem<T>> copy(Span<T> src, u64 st, u64 padding = 0) {
    return copy(src, st, src.size - st, padding);
}

template<typename T>
Block<ContainerElem<T>> copy(Span<T> src, u64 padding = 0) {
    return copy(src, 0, src.size, padding);
}

template<typename T>
Block<T> copy(const Block<T>& src, u64 padding = 0) {
    return copy(span(src), 0, src.size, padding);
}

template<typename T, u64 N>
Block<T> copy(const Array<T, N>& src, u64 padding = 0) {
    return copy(span(src), 0, src.size, padding);
}

template<typename T, u64... Dims>
Block<T> copy(const ArrayND<T, Dims...>& src, u64 padding = 0) {
    return copy(span(src.flat()), 0, src.flat().size, padding);
}

template<typename T, u64... Dims>
Block<ContainerElem<T>> copy(ArrayNDView<T, Dims...> src, u64 padding = 0) {
    return copy(span(src.flat()), 0, src.flat().size, padding);
}

} // namespace containers
