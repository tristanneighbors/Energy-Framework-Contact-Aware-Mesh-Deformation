#pragma once

/**
 * @file geometry_query.cuh
 * @brief Header-only ray, triangle, closest-point, and AABB primitives.
 */

#include "geometry_types.cuh"

namespace geom {

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 component(vec3 v, i32 axis) {
	return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void setComponent(vec3 *v, i32 axis, f32 value) {
	if (axis == 0) {
		v->x = value;
	} else if (axis == 1) {
		v->y = value;
	} else {
		v->z = value;
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 minVec(vec3 a, vec3 b) {
	return vec3(
		sprMin<f32>(a.x, b.x),
		sprMin<f32>(a.y, b.y),
		sprMin<f32>(a.z, b.z));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 maxVec(vec3 a, vec3 b) {
	return vec3(
		sprMax<f32>(a.x, b.x),
		sprMax<f32>(a.y, b.y),
		sprMax<f32>(a.z, b.z));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
Aabb emptyAabb() {
	Aabb out;
	out.lower = vec3(F32_MAX, F32_MAX, F32_MAX);
	out.upper = vec3(F32_LOWEST, F32_LOWEST, F32_LOWEST);
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool isValidAabb(Aabb bounds) {
	return
		bounds.lower.x <= bounds.upper.x &&
		bounds.lower.y <= bounds.upper.y &&
		bounds.lower.z <= bounds.upper.z;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void growAabbPoint(Aabb *boundsInOut, vec3 p) {
	boundsInOut->lower = minVec(boundsInOut->lower, p);
	boundsInOut->upper = maxVec(boundsInOut->upper, p);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void growAabb(Aabb *boundsInOut, Aabb other) {
	boundsInOut->lower = minVec(boundsInOut->lower, other.lower);
	boundsInOut->upper = maxVec(boundsInOut->upper, other.upper);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
Aabb unionAabb(Aabb a, Aabb b) {
	Aabb out;
	out.lower = minVec(a.lower, b.lower);
	out.upper = maxVec(a.upper, b.upper);
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 aabbCentroid(Aabb bounds) {
	return (bounds.lower + bounds.upper)*0.5f;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 aabbExtent(Aabb bounds) {
	return bounds.upper - bounds.lower;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
i32 largestAxis(vec3 v) {
	if (v.x >= v.y && v.x >= v.z) {
		return 0;
	}
	return v.y >= v.z ? 1 : 2;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
Aabb triangleAabb(vec3 a, vec3 b, vec3 c) {
	Aabb out = emptyAabb();
	growAabbPoint(&out, a);
	growAabbPoint(&out, b);
	growAabbPoint(&out, c);
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 sqDistancePointAabb(vec3 p, Aabb bounds) {
	f32 d2 = 0.0f;

	SPR_UNROLL
	for (i32 axis = 0; axis < 3; ++axis) {
		const f32 x = component(p, axis);
		const f32 lo = component(bounds.lower, axis);
		const f32 hi = component(bounds.upper, axis);

		f32 d = 0.0f;
		if (x < lo) {
			d = lo - x;
		} else if (x > hi) {
			d = x - hi;
		}
		d2 += d*d;
	}

	return d2;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
Ray makeRay(vec3 origin, vec3 dir, f32 tMin, f32 tMax) {
	Ray out;
	out.origin = origin;
	out.dir = dir;
	out.tMin = tMin;
	out.tMax = tMax;
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 rayPoint(Ray ray, f32 t) {
	return ray.origin + t*ray.dir;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool intersectRayAabb(f32 *tNearOut, f32 *tFarOut, Ray ray, Aabb bounds) {
	f32 tNear = ray.tMin;
	f32 tFar = ray.tMax;

	SPR_UNROLL
	for (i32 axis = 0; axis < 3; ++axis) {
		const f32 o = component(ray.origin, axis);
		const f32 d = component(ray.dir, axis);
		const f32 lo = component(bounds.lower, axis);
		const f32 hi = component(bounds.upper, axis);

		if (sprAbs<f32>(d) <= 1.0e-12f) {
			if (o < lo || o > hi) {
				return false;
			}
			continue;
		}

		const f32 invD = 1.0f/d;
		f32 t0 = (lo - o)*invD;
		f32 t1 = (hi - o)*invD;
		if (t0 > t1) {
			const f32 tmp = t0;
			t0 = t1;
			t1 = tmp;
		}

		tNear = sprMax<f32>(tNear, t0);
		tFar = sprMin<f32>(tFar, t1);
		if (tNear > tFar) {
			return false;
		}
	}

	if (tNearOut != nullptr) {
		*tNearOut = tNear;
	}
	if (tFarOut != nullptr) {
		*tFarOut = tFar;
	}
	return true;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 normalFromAreaVector(vec3 n) {
	const f32 n2 = sqNorm(n);
	if (!(n2 > 1.0e-30f) || !sprIsFinite(n2)) {
		return vec3(0.0f);
	}
	return n*sprRsqrt<f32>(n2);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 triangleNormal(vec3 a, vec3 b, vec3 c) {
	const vec3 n = cross(b - a, c - a);
	return normalFromAreaVector(n);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 triangleArea(vec3 a, vec3 b, vec3 c) {
	return 0.5f*length(cross(b - a, c - a));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 triangleCentroid(vec3 a, vec3 b, vec3 c) {
	return (a + b + c)*(1.0f/3.0f);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
ClosestPointResult closestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c, fid triIdx) {
	const vec3 ab = b - a;
	const vec3 ac = c - a;
	const vec3 ap = p - a;

	const f32 d1 = dot(ab, ap);
	const f32 d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		ClosestPointResult out;
		out.point = a;
		out.normal = triangleNormal(a, b, c);
		out.dist2 = sqNorm(p - a);
		out.u = 1.0f;
		out.v = 0.0f;
		out.w = 0.0f;
		out.triIdx = triIdx;
		return out;
	}

	const vec3 bp = p - b;
	const f32 d3 = dot(ab, bp);
	const f32 d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) {
		ClosestPointResult out;
		out.point = b;
		out.normal = triangleNormal(a, b, c);
		out.dist2 = sqNorm(p - b);
		out.u = 0.0f;
		out.v = 1.0f;
		out.w = 0.0f;
		out.triIdx = triIdx;
		return out;
	}

	const f32 vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		const f32 s = d1/(d1 - d3);
		const vec3 q = a + s*ab;
		ClosestPointResult out;
		out.point = q;
		out.normal = triangleNormal(a, b, c);
		out.dist2 = sqNorm(p - q);
		out.u = 1.0f - s;
		out.v = s;
		out.w = 0.0f;
		out.triIdx = triIdx;
		return out;
	}

	const vec3 cp = p - c;
	const f32 d5 = dot(ab, cp);
	const f32 d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) {
		ClosestPointResult out;
		out.point = c;
		out.normal = triangleNormal(a, b, c);
		out.dist2 = sqNorm(p - c);
		out.u = 0.0f;
		out.v = 0.0f;
		out.w = 1.0f;
		out.triIdx = triIdx;
		return out;
	}

	const f32 vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		const f32 s = d2/(d2 - d6);
		const vec3 q = a + s*ac;
		ClosestPointResult out;
		out.point = q;
		out.normal = triangleNormal(a, b, c);
		out.dist2 = sqNorm(p - q);
		out.u = 1.0f - s;
		out.v = 0.0f;
		out.w = s;
		out.triIdx = triIdx;
		return out;
	}

	const f32 va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		const f32 s = (d4 - d3)/((d4 - d3) + (d5 - d6));
		const vec3 q = b + s*(c - b);
		ClosestPointResult out;
		out.point = q;
		out.normal = triangleNormal(a, b, c);
		out.dist2 = sqNorm(p - q);
		out.u = 0.0f;
		out.v = 1.0f - s;
		out.w = s;
		out.triIdx = triIdx;
		return out;
	}

	const f32 denom = 1.0f/(va + vb + vc);
	const f32 v = vb*denom;
	const f32 w = vc*denom;
	const f32 u = 1.0f - v - w;
	const vec3 q = u*a + v*b + w*c;

	ClosestPointResult out;
	out.point = q;
	out.normal = triangleNormal(a, b, c);
	out.dist2 = sqNorm(p - q);
	out.u = u;
	out.v = v;
	out.w = w;
	out.triIdx = triIdx;
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool intersectRayTriangle(RayTriHit *hitOut, Ray ray, vec3 a, vec3 b, vec3 c, fid triIdx) {
	const f32 eps = 1.0e-7f;
	const vec3 e1 = b - a;
	const vec3 e2 = c - a;
	const vec3 p = cross(ray.dir, e2);
	const f32 det = dot(e1, p);

	if (sprAbs<f32>(det) <= eps) {
		return false;
	}

	const f32 invDet = 1.0f/det;
	const vec3 tvec = ray.origin - a;
	const f32 v = dot(tvec, p)*invDet;
	if (v < -eps || v > 1.0f + eps) {
		return false;
	}

	const vec3 q = cross(tvec, e1);
	const f32 w = dot(ray.dir, q)*invDet;
	if (w < -eps || v + w > 1.0f + eps) {
		return false;
	}

	const f32 t = dot(e2, q)*invDet;
	if (t < ray.tMin || t > ray.tMax) {
		return false;
	}

	if (hitOut != nullptr) {
		hitOut->t = t;
		hitOut->u = 1.0f - v - w;
		hitOut->v = v;
		hitOut->w = w;
		hitOut->triIdx = triIdx;
	}
	return true;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 defaultSignRayDir() {
	// Fixed non-axis direction. This reduces edge/vertex coincidences for parity tests.
	return normal(vec3(0.8111071f, 0.3244428f, 0.4866643f));
}

} // namespace geom
