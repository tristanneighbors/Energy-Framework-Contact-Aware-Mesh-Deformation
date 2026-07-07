#pragma once

/**
 * @file random.cuh
 * @brief Lightweight deterministic random sampling helpers.
 *
 * This generator is convenient for tests, sampling, and CUDA-friendly utility
 * code. It is not intended for cryptography or high-quality statistical work.
 */

#include "spr_global_include.h"
#include "config.hpp"
#include "vec.cuh"
#include "quat.cuh"
#include "rot.cuh"

/** Small f32-oriented deterministic generator with vector and rotation samplers. */
struct RandomGenerator {
    u32 hash;

    SPR_CUDA_HOST_DEVICE_INLINE explicit RandomGenerator(u32 seed) : hash(seed) {}

    SPR_CUDA_HOST_DEVICE_INLINE static u32 murmurHash3(u32 key) {
        const u32 c1 = 0xcc9e2d51u;
        const u32 c2 = 0x1b873593u;
        const u32 r1 = 15u;
        const u32 r2 = 13u;
        const u32 m = 5u;
        const u32 n = 0xe6546b64u;

        u32 h = key;

        key *= c1;
        key = (key << r1) | (key >> (32u - r1));
        key *= c2;

        h ^= key;
        h = (h << r2) | (h >> (32u - r2));
        h = h * m + n;

        h ^= h >> 16u;
        h *= 0x85ebca6bu;
        h ^= h >> 13u;
        h *= 0xc2b2ae35u;
        h ^= h >> 16u;

        return h;
    }

    SPR_CUDA_HOST_DEVICE_INLINE u32 nextHash() {
        hash = murmurHash3(hash);
        return hash;
    }

    SPR_CUDA_HOST_DEVICE_INLINE f32 nextf32(f32 low = 0.0f, f32 high = 1.0f) {
        const u32 h = nextHash();
        const f64 real01 = ((f64)h) / ((f64)4294967295.0);
        return (f32)(real01 * (high - low) + low);
    }

    SPR_CUDA_HOST_DEVICE_INLINE vec2 nextVec2(vec2 low = vec2(0.0f), vec2 high = vec2(1.0f)) {
        return vec2(nextf32(low.x, high.x), nextf32(low.y, high.y));
    }

    SPR_CUDA_HOST_DEVICE_INLINE vec3 nextVec3(vec3 low = vec3(0.0f), vec3 high = vec3(1.0f)) {
        return vec3(nextf32(low.x, high.x), nextf32(low.y, high.y), nextf32(low.z, high.z));
    }

    SPR_CUDA_HOST_DEVICE_INLINE vec4 nextVec4(vec4 low = vec4(0.0f), vec4 high = vec4(1.0f)) {
        return vec4(nextf32(low.x, high.x), nextf32(low.y, high.y), nextf32(low.z, high.z), nextf32(low.w, high.w));
    }

    SPR_CUDA_HOST_DEVICE_INLINE vec3 nextPointOnSphere() {
        const vec2 u = nextVec2(vec2(0.0f), vec2(1.0f));
        const f32 phi = sprAcosf(2.0f * u.x - 1.0f) - (f32)SPR_PI * 0.5f;
        const f32 lambda = 2.0f * (f32)SPR_PI * u.y;

        return vec3(sprCosf(phi) * sprCosf(lambda), sprCosf(phi) * sprSinf(lambda), sprSinf(phi));
    }

    SPR_CUDA_HOST_DEVICE_INLINE quat nextUnitQuaternion() {
        // Shoemake-style branch-light sampler; returns a unit quaternion directly.
        const f32 u1 = nextf32();
        const f32 u2 = nextf32();
        const f32 u3 = nextf32();
        const f32 a = sprSqrtf(1.0f - u1);
        const f32 b = sprSqrtf(u1);
        const f32 theta1 = 2.0f * (f32)SPR_PI * u2;
        const f32 theta2 = 2.0f * (f32)SPR_PI * u3;

        const f32 x = a * sprSinf(theta1);
        const f32 y = a * sprCosf(theta1);
        const f32 z = b * sprSinf(theta2);
        const f32 w = b * sprCosf(theta2);

        return quat(w, x, y, z);
    }

    SPR_CUDA_HOST_DEVICE_INLINE rot3 nextRot3() {
        return rot3FromUnitQuatUnchecked(nextUnitQuaternion());
    }
};
