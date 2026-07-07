#pragma once

/**
 * @file adsdf_render.cuh
 * @brief Minimal ADSDF ray marcher and PPM output helper.
 *
 * The marcher is intentionally conservative: filtered algebraic SDFs are not
 * guaranteed lower bounds, so step size is scaled and clamped rather than using
 * unrestricted sphere-tracing steps.
 */

#include "adsdf/adsdf_types.cuh"
#include "adsdf/adsdf_query.cuh"

#include "dbg/cuda_utils.cuh"

#include <cmath>
#include <cstdio>

namespace adsdf {

#if defined(__CUDACC__)

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
u32 packRgba8(f32 r, f32 g, f32 b, f32 a) {
	const u32 ri = u32(sprClampf(r, 0.0f, 1.0f)*255.0f + 0.5f);
	const u32 gi = u32(sprClampf(g, 0.0f, 1.0f)*255.0f + 0.5f);
	const u32 bi = u32(sprClampf(b, 0.0f, 1.0f)*255.0f + 0.5f);
	const u32 ai = u32(sprClampf(a, 0.0f, 1.0f)*255.0f + 0.5f);
	return ri | (gi << 8) | (bi << 16) | (ai << 24);
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
bool intersectAabb(
	f32 *tMinOut,
	f32 *tMaxOut,
	vec3 rayOrigin,
	vec3 rayDir,
	vec3 boxMin,
	vec3 boxMax) {
	f32 tMin = 0.0f;
	f32 tMax = F32_MAX;

	for (i32 axis = 0; axis < 3; ++axis) {
		const f32 origin = rayOrigin[axis];
		const f32 dir = rayDir[axis];
		const f32 minv = boxMin[axis];
		const f32 maxv = boxMax[axis];

		if (fabsf(dir) < 1.0e-8f) {
			if (origin < minv || origin > maxv) {
				return false;
			}
			continue;
		}

		const f32 invDir = 1.0f/dir;
		f32 t0 = (minv - origin)*invDir;
		f32 t1 = (maxv - origin)*invDir;
		if (t0 > t1) {
			const f32 tmp = t0;
			t0 = t1;
			t1 = tmp;
		}

		tMin = sprFmaxf(tMin, t0);
		tMax = sprFminf(tMax, t1);
		if (tMax < tMin) {
			return false;
		}
	}

	*tMinOut = tMin;
	*tMaxOut = tMax;
	return true;
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec3 shadeNormal(AdsdfView field, vec3 hitPos, vec3 grad, AdsdfRayMarchParams params) {
	vec3 n = grad;
	const f32 nSq = sqNorm(n);
	if (nSq < 1.0e-12f || !sprIsFinite(nSq)) {
		n = queryNormalFiniteDiff(field, hitPos, params.normalEps);
	} else {
		n = normal(n);
	}

	vec3 lightDir = params.lightDir;
	if (sqNorm(lightDir) < 1.0e-12f) {
		lightDir = vec3(0.5f, 0.7f, 1.0f);
	}
	lightDir = normal(lightDir);

	const f32 ndotl = sprFmaxf(dot(n, lightDir), 0.0f);
	const f32 diffuse = 0.15f + 0.85f*ndotl;
	return vec3(diffuse);
}

static __global__
void render(
	u32 *pixelsOut_d,
	AdsdfView field,
	AdsdfCamera camera,
	AdsdfRayMarchParams params) {
	const i32 numPixels = params.imageWidth*params.imageHeight;

	for (i32 pixelIdx = dbg::globalThreadIdx1d(); pixelIdx < numPixels; pixelIdx += dbg::globalThreadStride1d()) {
		const i32 px = pixelIdx % params.imageWidth;
		const i32 py = pixelIdx / params.imageWidth;

		const f32 sx = 2.0f*(f32(px) + 0.5f)/f32(params.imageWidth) - 1.0f;
		const f32 sy = 1.0f - 2.0f*(f32(py) + 0.5f)/f32(params.imageHeight);
		const vec3 rayDir = normal(camera.dir + camera.right*(sx*camera.tanHalfFovY*camera.aspect) + camera.up*(sy*camera.tanHalfFovY));

		f32 t0;
		f32 t1;
		if (!intersectAabb(&t0, &t1, camera.pos, rayDir, field.desc.domainMin, field.desc.domainMax)) {
			pixelsOut_d[pixelIdx] = packRgba8(0.02f, 0.025f, 0.03f, 1.0f);
			continue;
		}

		t0 = sprFmaxf(t0, 0.0f);
		t1 = sprFminf(t1, params.tMax);

		f32 t = t0;
		f32 prevT = t;
		f32 prevValue = F32_MAX;
		bool hit = false;
		vec3 hitPos(0.0f);
		vec3 hitGrad(0.0f);

		for (i32 stepIdx = 0; stepIdx < params.maxSteps && t <= t1; ++stepIdx) {
			const vec3 pos = camera.pos + rayDir*t;
			const AdsdfQueryResult qr = query(field, pos);
			const f32 d = qr.value;

			if (d < 0.0f && prevValue > 0.0f) {
				f32 lo = prevT;
				f32 hi = t;
				for (i32 refine = 0; refine < 8; ++refine) {
					const f32 mid = 0.5f*(lo + hi);
					const f32 midValue = queryValue(field, camera.pos + rayDir*mid);
					if (midValue > 0.0f) {
						lo = mid;
					} else {
						hi = mid;
					}
				}

				t = 0.5f*(lo + hi);
				hitPos = camera.pos + rayDir*t;
				hitGrad = query(field, hitPos).grad;
				hit = true;
				break;
			}

			if (d >= 0.0f && d < params.hitEps) {
				hit = true;
				hitPos = pos;
				hitGrad = qr.grad;
				break;
			}

			prevT = t;
			prevValue = d;

			const f32 step = sprClampf(params.stepScale*sprFmaxf(d, params.minStep), params.minStep, params.maxStep);
			t += step;
		}

		if (!hit) {
			pixelsOut_d[pixelIdx] = packRgba8(0.02f, 0.025f, 0.03f, 1.0f);
			continue;
		}

		const vec3 color = shadeNormal(field, hitPos, hitGrad, params);
		pixelsOut_d[pixelIdx] = packRgba8(color.x, color.y, color.z, 1.0f);
	}
}

inline void launchRender(
	u32 *pixelsOut_d,
	AdsdfView field,
	AdsdfCamera camera,
	AdsdfRayMarchParams params,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(pixelsOut_d != nullptr);

	const i32 numPixels = params.imageWidth*params.imageHeight;
	const i32 threadsPerBlock = ADSDF_DEFAULT_THREADS_PER_BLOCK;
	const i32 numBlocks = dbg::numBlocks(numPixels, threadsPerBlock);

	render<<<numBlocks, threadsPerBlock, 0, stream>>>(
		pixelsOut_d,
		field,
		camera,
		params);
	CUDA_LAUNCH_CHECK();
}

#endif

[[nodiscard]] inline bool writePpm(
	const char *path,
	const u32 *pixels_h,
	i32 width,
	i32 height) {
	if (path == nullptr || pixels_h == nullptr || width <= 0 || height <= 0) {
		return false;
	}

	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	std::fprintf(file, "P6\n%d %d\n255\n", width, height);
	for (i32 i = 0; i < width*height; ++i) {
		const u32 pixel = pixels_h[i];
		const u8 rgb[3] = {
			u8(pixel & 0xffu),
			u8((pixel >> 8) & 0xffu),
			u8((pixel >> 16) & 0xffu),
		};
		std::fwrite(rgb, 1, 3, file);
	}

	std::fclose(file);
	return true;
}

} // namespace adsdf
