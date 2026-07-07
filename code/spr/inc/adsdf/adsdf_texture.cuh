#pragma once

/**
 * @file adsdf_texture.cuh
 * @brief CUDA texture storage management for ADSDF coefficient groups.
 *
 * Each coefficient group is uploaded as its own 3D `float4` texture. Hardware
 * linear filtering can then interpolate coefficients directly, which is valid
 * because construction stores all node polynomials in the same global basis.
 */

#include "adsdf/adsdf_types.cuh"
#include "adsdf/adsdf_query.cuh"

#include "dbg/cuda_utils.cuh"

namespace adsdf {

static inline void clearTextureGrid(AdsdfTextureGrid *grid) {
	for (i32 i = 0; i < ADSDF_MAX_COEFF_GROUPS; ++i) {
		grid->coeffArrays[i] = nullptr;
		grid->coeffTextures[i] = 0;
	}
}

inline void allocTextureGrid(AdsdfTextureGrid *gridOut, const AdsdfDesc *desc) {
	SPR_ASSERT(gridOut != nullptr);
	SPR_ASSERT(desc != nullptr);

	gridOut->desc = *desc;
	clearTextureGrid(gridOut);

	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	const cudaExtent extent = make_cudaExtent(
		size_t(desc->numX),
		size_t(desc->numY),
		size_t(desc->numZ));

	for (i32 groupIdx = 0; groupIdx < desc->numCoeffGroups; ++groupIdx) {
		CUDA_CHECK(cudaMalloc3DArray(&gridOut->coeffArrays[groupIdx], &channelDesc, extent));

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = gridOut->coeffArrays[groupIdx];

		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = desc->filterKind == AdsdfFilterKind::Nearest ?
			cudaFilterModePoint : cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA_CHECK(cudaCreateTextureObject(
			&gridOut->coeffTextures[groupIdx],
			&resDesc,
			&texDesc,
			nullptr));
	}
}

inline void uploadTextureGrid(AdsdfTextureGrid *grid, const AdsdfLinearGrid *src) {
	SPR_ASSERT(grid != nullptr);
	SPR_ASSERT(src != nullptr);
	SPR_ASSERT(src->coeffGroups_d != nullptr);
	SPR_ASSERT(grid->desc.numCoeffGroups == src->desc.numCoeffGroups);

	const AdsdfDesc *desc = &src->desc;
	const size_t nodes = numNodes(desc);
	const cudaExtent extent = make_cudaExtent(
		size_t(desc->numX),
		size_t(desc->numY),
		size_t(desc->numZ));

	for (i32 groupIdx = 0; groupIdx < desc->numCoeffGroups; ++groupIdx) {
		cudaMemcpy3DParms copy = {};
		copy.srcPtr = make_cudaPitchedPtr(
			src->coeffGroups_d + size_t(groupIdx)*nodes,
			size_t(desc->numX)*sizeof(vec4),
			size_t(desc->numX),
			size_t(desc->numY));
		copy.dstArray = grid->coeffArrays[groupIdx];
		copy.extent = extent;
		copy.kind = cudaMemcpyDeviceToDevice;
		CUDA_CHECK(cudaMemcpy3D(&copy));
	}
}

inline void freeTextureGrid(AdsdfTextureGrid *grid) {
	if (grid == nullptr) {
		return;
	}

	for (i32 groupIdx = 0; groupIdx < ADSDF_MAX_COEFF_GROUPS; ++groupIdx) {
		if (grid->coeffTextures[groupIdx] != 0) {
			CUDA_CHECK(cudaDestroyTextureObject(grid->coeffTextures[groupIdx]));
			grid->coeffTextures[groupIdx] = 0;
		}

		if (grid->coeffArrays[groupIdx] != nullptr) {
			CUDA_CHECK(cudaFreeArray(grid->coeffArrays[groupIdx]));
			grid->coeffArrays[groupIdx] = nullptr;
		}
	}

	grid->desc = {};
}

inline AdsdfView makeView(const AdsdfTextureGrid *grid) {
	AdsdfView view = {};
	view.desc = grid->desc;
	view.storageKind = AdsdfStorageKind::Texture3D;
	view.coeffGroups_d = nullptr;

	for (i32 i = 0; i < ADSDF_MAX_COEFF_GROUPS; ++i) {
		view.coeffTextures[i] = grid->coeffTextures[i];
	}

	return view;
}

} // namespace adsdf
