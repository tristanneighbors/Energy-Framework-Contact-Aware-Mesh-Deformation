#pragma once

/**
 * @file adsdf_sample.cuh
 * @brief CUDA point sampling helpers for ADSDF fields.
 */

#include "adsdf/adsdf_query.cuh"

#include "dbg/cuda_utils.cuh"

#include <vector>

namespace adsdf {

struct AdsdfPointSample {
	f32 phi;
	vec3 grad;
};

struct GpuAdsdfSampler {
	vec3 *positions_d;
	AdsdfPointSample *samples_d;
	i32 capacity;
	bool ownsMemory;
};

#if defined(__CUDACC__)

inline void initGpuAdsdfSampler(GpuAdsdfSampler *samplerOut) {
	if (samplerOut == nullptr) {
		return;
	}

	samplerOut->positions_d = nullptr;
	samplerOut->samples_d = nullptr;
	samplerOut->capacity = 0;
	samplerOut->ownsMemory = false;
}

inline void freeGpuAdsdfSampler(GpuAdsdfSampler *sampler) {
	if (sampler == nullptr) {
		return;
	}

	if (sampler->ownsMemory) {
		if (sampler->positions_d != nullptr) { CUDA_CHECK(cudaFree(sampler->positions_d)); }
		if (sampler->samples_d != nullptr) { CUDA_CHECK(cudaFree(sampler->samples_d)); }
	}

	initGpuAdsdfSampler(sampler);
}

inline void reserveGpuAdsdfSampler(GpuAdsdfSampler *sampler, i32 capacity) {
	if (sampler == nullptr || capacity <= sampler->capacity) {
		return;
	}

	freeGpuAdsdfSampler(sampler);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sampler->positions_d), size_t(capacity)*sizeof(vec3)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sampler->samples_d), size_t(capacity)*sizeof(AdsdfPointSample)));
	sampler->capacity = capacity;
	sampler->ownsMemory = true;
}

static __global__
void sampleAdsdfPoints(
	AdsdfPointSample *samplesOut_d,
	const vec3 *positions_d,
	i32 numPositions,
	AdsdfView field,
	f32 normalEps) {
	for (i32 idx = dbg::globalThreadIdx1d(); idx < numPositions; idx += dbg::globalThreadStride1d()) {
		const vec3 p = positions_d[idx];
		const AdsdfQueryResult qr = query(field, p);
		vec3 grad = qr.grad;
		const f32 g2 = sqNorm(grad);

		if (g2 <= 1.0e-10f || !sprIsFinite(g2)) {
			grad = queryNormalFiniteDiff(field, p, normalEps);
		} else {
			grad = grad*sprRsqrt<f32>(g2);
		}

		samplesOut_d[idx].phi = qr.value;
		samplesOut_d[idx].grad = grad;
	}
}

inline void launchSampleAdsdfPoints(
	AdsdfPointSample *samplesOut_d,
	const vec3 *positions_d,
	i32 numPositions,
	AdsdfView field,
	f32 normalEps,
	i32 threadsPerBlock = ADSDF_DEFAULT_THREADS_PER_BLOCK,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(samplesOut_d != nullptr);
	SPR_ASSERT(positions_d != nullptr);

	if (numPositions <= 0) {
		return;
	}

	const i32 numBlocks = dbg::numBlocks(numPositions, threadsPerBlock);
	sampleAdsdfPoints<<<numBlocks, threadsPerBlock, 0, stream>>>(
		samplesOut_d,
		positions_d,
		numPositions,
		field,
		normalEps);
	CUDA_LAUNCH_CHECK();
}

inline const std::vector<AdsdfPointSample> &sampleAdsdfPointsBlocking(
	std::vector<AdsdfPointSample> *samplesOut_h,
	GpuAdsdfSampler *sampler,
	AdsdfView field,
	const std::vector<vec3> &positions,
	f32 normalEps,
	i32 threadsPerBlock = ADSDF_DEFAULT_THREADS_PER_BLOCK,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(samplesOut_h != nullptr);
	SPR_ASSERT(sampler != nullptr);

	const i32 n = i32(positions.size());
	samplesOut_h->resize(size_t(n));
	if (n <= 0) {
		return *samplesOut_h;
	}

	reserveGpuAdsdfSampler(sampler, n);
	CUDA_CHECK(cudaMemcpyAsync(
		sampler->positions_d,
		positions.data(),
		size_t(n)*sizeof(vec3),
		cudaMemcpyHostToDevice,
		stream));

	launchSampleAdsdfPoints(
		sampler->samples_d,
		sampler->positions_d,
		n,
		field,
		normalEps,
		threadsPerBlock,
		stream);

	CUDA_CHECK(cudaMemcpyAsync(
		samplesOut_h->data(),
		sampler->samples_d,
		size_t(n)*sizeof(AdsdfPointSample),
		cudaMemcpyDeviceToHost,
		stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return *samplesOut_h;
}

#endif

} // namespace adsdf
