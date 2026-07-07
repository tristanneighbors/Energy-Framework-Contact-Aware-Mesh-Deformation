#pragma once

/**
 * @file geometry_sdf.cuh
 * @brief Mesh signed-distance samples and CUDA sampling kernels.
 */

#include "geometry_bvh.cuh"

#include <vector>

namespace geom {

struct MeshSdfSample {
	f32 phi;
	vec3 grad;
	vec3 closestPoint;
	vec3 closestNormal;
	f32 dist;
	fid triIdx;
};

struct GpuMeshSdfSampler {
	vec3 *positions_d;
	MeshSdfSample *samples_d;
	i32 capacity;
	bool ownsMemory;
};

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
MeshSdfSample makeInvalidMeshSdfSample() {
	MeshSdfSample out = {};
	out.phi = F32_MAX;
	out.grad = vec3(0.0f);
	out.closestPoint = vec3(0.0f);
	out.closestNormal = vec3(0.0f);
	out.dist = F32_MAX;
	out.triIdx = GEOM_INVALID_IDX;
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 meshSdfSign(MeshView mesh, BvhView bvh, vec3 p, ClosestPointResult closest, MeshSignMethod signMethod) {
	if (signMethod == MeshSignMethod::Unsigned) {
		return 1.0f;
	}

	if (signMethod == MeshSignMethod::NormalPseudo) {
		return dot(p - closest.point, closest.normal) < 0.0f ? -1.0f : 1.0f;
	}

	const Ray ray = makeRay(p, defaultSignRayDir(), 1.0e-4f, F32_MAX);
	const i32 count = countRayIntersectionsBvh(mesh, bvh, ray, GEOM_RAY_COUNT_MAX_HITS);
	return (count & 1) ? -1.0f : 1.0f;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
MeshSdfSample sampleMeshSdf(MeshView mesh, BvhView bvh, vec3 p, MeshSignMethod signMethod) {
	ClosestPointResult closest;
	if (!closestPointBvh(&closest, mesh, bvh, p)) {
		return makeInvalidMeshSdfSample();
	}

	const f32 dist = sprSqrt<f32>(closest.dist2);
	const f32 sign = meshSdfSign(mesh, bvh, p, closest, signMethod);

	vec3 grad = closest.normal;
	if (dist > 1.0e-7f && sprIsFinite(dist)) {
		grad = normal(p - closest.point);
		if (sign < 0.0f) {
			grad = -grad;
		}
	}

	const f32 grad2 = sqNorm(grad);
	if (grad2 < 1.0e-8f || !sprIsFinite(grad2)) {
		grad = vec3(0.0f, 0.0f, 1.0f);
	}

	MeshSdfSample out = {};
	out.phi = sign*dist;
	out.grad = grad;
	out.closestPoint = closest.point;
	out.closestNormal = closest.normal;
	out.dist = dist;
	out.triIdx = closest.triIdx;
	return out;
}

#if defined(__CUDACC__)

inline void initGpuMeshSdfSampler(GpuMeshSdfSampler *samplerOut) {
	if (samplerOut == nullptr) {
		return;
	}

	samplerOut->positions_d = nullptr;
	samplerOut->samples_d = nullptr;
	samplerOut->capacity = 0;
	samplerOut->ownsMemory = false;
}

inline void freeGpuMeshSdfSampler(GpuMeshSdfSampler *sampler) {
	if (sampler == nullptr) {
		return;
	}

	if (sampler->ownsMemory) {
		if (sampler->positions_d != nullptr) { CUDA_CHECK(cudaFree(sampler->positions_d)); }
		if (sampler->samples_d != nullptr) { CUDA_CHECK(cudaFree(sampler->samples_d)); }
	}

	initGpuMeshSdfSampler(sampler);
}

inline void reserveGpuMeshSdfSampler(GpuMeshSdfSampler *sampler, i32 capacity) {
	if (sampler == nullptr || capacity <= sampler->capacity) {
		return;
	}

	freeGpuMeshSdfSampler(sampler);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sampler->positions_d), size_t(capacity)*sizeof(vec3)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sampler->samples_d), size_t(capacity)*sizeof(MeshSdfSample)));
	sampler->capacity = capacity;
	sampler->ownsMemory = true;
}

static __global__
void sampleMeshSdfPoints(
	MeshSdfSample *samplesOut_d,
	const vec3 *positions_d,
	i32 numPositions,
	MeshView mesh,
	BvhView bvh,
	MeshSignMethod signMethod) {
	for (i32 idx = dbg::globalThreadIdx1d(); idx < numPositions; idx += dbg::globalThreadStride1d()) {
		samplesOut_d[idx] = sampleMeshSdf(mesh, bvh, positions_d[idx], signMethod);
	}
}

inline void launchSampleMeshSdfPoints(
	MeshSdfSample *samplesOut_d,
	const vec3 *positions_d,
	i32 numPositions,
	MeshView mesh,
	BvhView bvh,
	MeshSignMethod signMethod = MeshSignMethod::RayParity,
	i32 threadsPerBlock = 256,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(samplesOut_d != nullptr);
	SPR_ASSERT(positions_d != nullptr);

	if (numPositions <= 0) {
		return;
	}

	const i32 numBlocks = dbg::numBlocks(numPositions, threadsPerBlock);
	sampleMeshSdfPoints<<<numBlocks, threadsPerBlock, 0, stream>>>(
		samplesOut_d,
		positions_d,
		numPositions,
		mesh,
		bvh,
		signMethod);
	CUDA_LAUNCH_CHECK();
}

inline const std::vector<MeshSdfSample> &sampleMeshSdfPointsBlocking(
	std::vector<MeshSdfSample> *samplesOut_h,
	GpuMeshSdfSampler *sampler,
	MeshView mesh,
	BvhView bvh,
	const std::vector<vec3> &positions,
	MeshSignMethod signMethod = MeshSignMethod::RayParity,
	i32 threadsPerBlock = 256,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(samplesOut_h != nullptr);
	SPR_ASSERT(sampler != nullptr);

	const i32 n = i32(positions.size());
	samplesOut_h->resize(size_t(n));
	if (n <= 0) {
		return *samplesOut_h;
	}

	reserveGpuMeshSdfSampler(sampler, n);
	CUDA_CHECK(cudaMemcpyAsync(
		sampler->positions_d,
		positions.data(),
		size_t(n)*sizeof(vec3),
		cudaMemcpyHostToDevice,
		stream));

	launchSampleMeshSdfPoints(
		sampler->samples_d,
		sampler->positions_d,
		n,
		mesh,
		bvh,
		signMethod,
		threadsPerBlock,
		stream);

	CUDA_CHECK(cudaMemcpyAsync(
		samplesOut_h->data(),
		sampler->samples_d,
		size_t(n)*sizeof(MeshSdfSample),
		cudaMemcpyDeviceToHost,
		stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return *samplesOut_h;
}

#endif

} // namespace geom
