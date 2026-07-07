#pragma once

/**
 * @file sdf_sources.cuh
 * @brief Simple procedural SDF evaluators used by the ADSDF builder.
 *
 * These sources are intentionally compact parameter packs for early construction
 * tests. More complicated inputs, such as dense grids or meshes, should get
 * separate source types instead of expanding `ProceduralSdf` indefinitely.
 */

#include "adsdf/adsdf_types.cuh"

namespace adsdf {

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 absVec(vec3 v) {
	return vec3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 maxVec(vec3 a, vec3 b) {
	return vec3(
		sprFmaxf(a.x, b.x),
		sprFmaxf(a.y, b.y),
		sprFmaxf(a.z, b.z));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 max3(f32 a, f32 b, f32 c) {
	return sprFmaxf(a, sprFmaxf(b, c));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalSphereSdf(const ProceduralSdf *source, vec3 x) {
	const vec3 center(source->a.x, source->a.y, source->a.z);
	const f32 radius = source->a.w;
	return length(x - center) - radius;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalBoxSdf(const ProceduralSdf *source, vec3 x) {
	const vec3 center(source->a.x, source->a.y, source->a.z);
	const vec3 halfExtent(source->b.x, source->b.y, source->b.z);
	const vec3 q = absVec(x - center) - halfExtent;
	const vec3 outside = maxVec(q, vec3(0.0f));
	const f32 inside = sprFminf(max3(q.x, q.y, q.z), 0.0f);
	return length(outside) + inside;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalTorusSdf(const ProceduralSdf *source, vec3 x) {
	const vec3 center(source->a.x, source->a.y, source->a.z);
	const f32 majorRadius = source->a.w;
	const f32 minorRadius = source->b.x;
	const vec3 p = x - center;
	const vec2 q(length(vec2(p.x, p.y)) - majorRadius, p.z);
	return length(q) - minorRadius;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalProceduralSdf(const ProceduralSdf *source, vec3 x) {
	switch (source->kind) {
		case ProceduralSdfKind::Sphere: return evalSphereSdf(source, x);
		case ProceduralSdfKind::Box: return evalBoxSdf(source, x);
		case ProceduralSdfKind::Torus: return evalTorusSdf(source, x);
		default: return F32_MAX;
	}
}

} // namespace adsdf
