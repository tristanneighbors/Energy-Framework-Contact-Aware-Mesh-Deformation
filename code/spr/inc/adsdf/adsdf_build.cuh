#pragma once

/**
 * @file adsdf_build.cuh
 * @brief Host setup and CUDA launches for procedural ADSDF construction.
 *
 * The current builder precomputes a local least-squares pseudoinverse on the
 * host, samples a procedural SDF around each lattice node on the GPU, fits local
 * coefficients, translates them into the global normalized basis, and writes
 * packed coefficient groups into linear device storage.
 */

#include "adsdf/adsdf_types.cuh"
#include "adsdf/sdf_sources.cuh"
#include "adsdf/adsdf_query.cuh"

#include "dbg/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace adsdf {

static inline bool invertSquareMatrix(
	std::vector<f64> *inverseOut,
	std::vector<f64> a,
	i32 n) {
	inverseOut->assign(size_t(n)*size_t(n), 0.0);

	for (i32 i = 0; i < n; ++i) {
		(*inverseOut)[size_t(i)*size_t(n) + size_t(i)] = 1.0;
	}

	for (i32 col = 0; col < n; ++col) {
		i32 pivotRow = col;
		f64 pivotAbs = std::fabs(a[size_t(col)*size_t(n) + size_t(col)]);

		for (i32 row = col + 1; row < n; ++row) {
			const f64 v = std::fabs(a[size_t(row)*size_t(n) + size_t(col)]);
			if (v > pivotAbs) {
				pivotAbs = v;
				pivotRow = row;
			}
		}

		if (pivotAbs < 1.0e-14) {
			return false;
		}

		if (pivotRow != col) {
			for (i32 j = 0; j < n; ++j) {
				std::swap(a[size_t(col)*size_t(n) + size_t(j)], a[size_t(pivotRow)*size_t(n) + size_t(j)]);
				std::swap((*inverseOut)[size_t(col)*size_t(n) + size_t(j)], (*inverseOut)[size_t(pivotRow)*size_t(n) + size_t(j)]);
			}
		}

		const f64 pivot = a[size_t(col)*size_t(n) + size_t(col)];
		const f64 invPivot = 1.0/pivot;

		for (i32 j = 0; j < n; ++j) {
			a[size_t(col)*size_t(n) + size_t(j)] *= invPivot;
			(*inverseOut)[size_t(col)*size_t(n) + size_t(j)] *= invPivot;
		}

		for (i32 row = 0; row < n; ++row) {
			if (row == col) {
				continue;
			}

			const f64 factor = a[size_t(row)*size_t(n) + size_t(col)];
			if (factor == 0.0) {
				continue;
			}

			for (i32 j = 0; j < n; ++j) {
				a[size_t(row)*size_t(n) + size_t(j)] -= factor*a[size_t(col)*size_t(n) + size_t(j)];
				(*inverseOut)[size_t(row)*size_t(n) + size_t(j)] -= factor*(*inverseOut)[size_t(col)*size_t(n) + size_t(j)];
			}
		}
	}

	return true;
}

static SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 fineOffsetQ(
	const AdsdfDesc *desc,
	const AdsdfLsqParams *params,
	i32 ox,
	i32 oy,
	i32 oz) {
	if (params->fineRadius <= 0) {
		return vec3(0.0f);
	}

	const f32 invRadius = 1.0f/f32(params->fineRadius);
	return vec3(
		params->fineExtent*desc->gridSpacingQ.x*f32(ox)*invRadius,
		params->fineExtent*desc->gridSpacingQ.y*f32(oy)*invRadius,
		params->fineExtent*desc->gridSpacingQ.z*f32(oz)*invRadius);
}

/** Initialize lattice, domain, derived spacing, degree, and filter metadata. */
inline void initDesc(
	AdsdfDesc *descOut,
	i32 numX,
	i32 numY,
	i32 numZ,
	i32 degree,
	vec3 domainMin,
	vec3 domainMax,
	AdsdfFilterKind filterKind = AdsdfFilterKind::Linear) {
	SPR_ASSERT(descOut != nullptr);
	SPR_ASSERT(numX >= 2 && numY >= 2 && numZ >= 2);
	SPR_ASSERT(degree >= 0 && degree <= ADSDF_MAX_DEGREE);

	AdsdfDesc desc = {};
	desc.numX = numX;
	desc.numY = numY;
	desc.numZ = numZ;
	desc.degree = degree;
	desc.numCoeffs = numCoeffs(degree);
	desc.numCoeffGroups = numCoeffGroups(degree);
	desc.domainMin = domainMin;
	desc.domainMax = domainMax;
	desc.domainCenter = (domainMin + domainMax)*0.5f;
	desc.domainHalfExtent = (domainMax - domainMin)*0.5f;
	desc.invDomainHalfExtent = vec3(
		1.0f/desc.domainHalfExtent.x,
		1.0f/desc.domainHalfExtent.y,
		1.0f/desc.domainHalfExtent.z);
	desc.gridSpacing = vec3(
		(domainMax.x - domainMin.x)/f32(numX - 1),
		(domainMax.y - domainMin.y)/f32(numY - 1),
		(domainMax.z - domainMin.z)/f32(numZ - 1));
	desc.invGridSpacing = vec3(
		1.0f/desc.gridSpacing.x,
		1.0f/desc.gridSpacing.y,
		1.0f/desc.gridSpacing.z);
	desc.gridSpacingQ = vec3(
		2.0f/f32(numX - 1),
		2.0f/f32(numY - 1),
		2.0f/f32(numZ - 1));
	desc.filterKind = filterKind;

	*descOut = desc;
}

inline void allocLinearGrid(AdsdfLinearGrid *gridOut, const AdsdfDesc *desc) {
	SPR_ASSERT(gridOut != nullptr);
	SPR_ASSERT(desc != nullptr);

	gridOut->desc = *desc;
	gridOut->coeffGroups_d = nullptr;

	const size_t size_bytes =
		size_t(desc->numCoeffGroups)*numNodes(desc)*sizeof(vec4);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gridOut->coeffGroups_d), size_bytes));
	CUDA_CHECK(cudaMemset(gridOut->coeffGroups_d, 0, size_bytes));
}

inline void freeLinearGrid(AdsdfLinearGrid *grid) {
	if (grid == nullptr) {
		return;
	}

	if (grid->coeffGroups_d != nullptr) {
		CUDA_CHECK(cudaFree(grid->coeffGroups_d));
		grid->coeffGroups_d = nullptr;
	}

	grid->desc = {};
}

inline AdsdfView makeView(const AdsdfLinearGrid *grid) {
	AdsdfView view = {};
	view.desc = grid->desc;
	view.storageKind = AdsdfStorageKind::Linear;
	view.coeffGroups_d = grid->coeffGroups_d;
	for (i32 i = 0; i < ADSDF_MAX_COEFF_GROUPS; ++i) {
		view.coeffTextures[i] = 0;
	}
	return view;
}

/** Compute row-major pseudoinverse P so local coefficients are `P * samples`. */
[[nodiscard]] inline bool computeLsqPseudoinverse(
	f32 *pinvOut_h,
	const AdsdfDesc *desc,
	const AdsdfLsqParams *params) {
	SPR_ASSERT(pinvOut_h != nullptr);
	SPR_ASSERT(desc != nullptr);
	SPR_ASSERT(params != nullptr);

	const i32 degree = desc->degree;
	const i32 k = desc->numCoeffs;
	const i32 h = params->fineRadius;
	const i32 fineDim = 2*h + 1;
	const i32 n = fineDim*fineDim*fineDim;

	if (degree < 0 || degree > ADSDF_MAX_DEGREE || k <= 0 || n < k) {
		return false;
	}

	std::vector<f64> x(size_t(n)*size_t(k), 0.0);

	i32 sampleIdx = 0;
	for (i32 oz = -h; oz <= h; ++oz) {
		for (i32 oy = -h; oy <= h; ++oy) {
			for (i32 ox = -h; ox <= h; ++ox) {
				const vec3 q = fineOffsetQ(desc, params, ox, oy, oz);
				for (i32 coeffIdx = 0; coeffIdx < k; ++coeffIdx) {
					x[size_t(sampleIdx)*size_t(k) + size_t(coeffIdx)] = f64(basisValue(coeffIdx, q));
				}
				++sampleIdx;
			}
		}
	}

	std::vector<f64> normalMat(size_t(k)*size_t(k), 0.0);
	for (i32 row = 0; row < k; ++row) {
		for (i32 col = 0; col < k; ++col) {
			f64 sum = 0.0;
			for (i32 sample = 0; sample < n; ++sample) {
				sum += x[size_t(sample)*size_t(k) + size_t(row)]*x[size_t(sample)*size_t(k) + size_t(col)];
			}
			if (row == col) {
				sum += f64(params->regularization);
			}
			normalMat[size_t(row)*size_t(k) + size_t(col)] = sum;
		}
	}

	std::vector<f64> normalInv;
	if (!invertSquareMatrix(&normalInv, normalMat, k)) {
		return false;
	}

	// P = (X^T X)^-1 X^T. Row-major K x N.
	for (i32 coeff = 0; coeff < k; ++coeff) {
		for (i32 sample = 0; sample < n; ++sample) {
			f64 sum = 0.0;
			for (i32 col = 0; col < k; ++col) {
				sum += normalInv[size_t(coeff)*size_t(k) + size_t(col)]*x[size_t(sample)*size_t(k) + size_t(col)];
			}
			pinvOut_h[size_t(coeff)*size_t(n) + size_t(sample)] = f32(sum);
		}
	}

	return true;
}

inline void allocAndUploadLsqPseudoinverse(
	f32 **pinvOut_d,
	const AdsdfDesc *desc,
	const AdsdfLsqParams *params) {
	SPR_ASSERT(pinvOut_d != nullptr);
	SPR_ASSERT(desc != nullptr);
	SPR_ASSERT(params != nullptr);

	const i32 h = params->fineRadius;
	const i32 fineDim = 2*h + 1;
	const i32 n = fineDim*fineDim*fineDim;
	const i32 k = desc->numCoeffs;
	const size_t size_bytes = size_t(k)*size_t(n)*sizeof(f32);

	std::vector<f32> pinv(size_t(k)*size_t(n));
	const bool ok = computeLsqPseudoinverse(pinv.data(), desc, params);
	SPR_ASSERT(ok);

	if (!ok) {
		*pinvOut_d = nullptr;
		return;
	}

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(pinvOut_d), size_bytes));
	CUDA_CHECK(cudaMemcpy(*pinvOut_d, pinv.data(), size_bytes, cudaMemcpyHostToDevice));
}

inline void freeLsqPseudoinverse(f32 **pinvInOut_d) {
	if (pinvInOut_d == nullptr || *pinvInOut_d == nullptr) {
		return;
	}

	CUDA_CHECK(cudaFree(*pinvInOut_d));
	*pinvInOut_d = nullptr;
}

#if defined(__CUDACC__)

static __global__
void buildProceduralLsq(
	AdsdfLinearGrid grid,
	const f32 *pinv_d,
	AdsdfLsqParams params,
	ProceduralSdf source) {
	const AdsdfDesc desc = grid.desc;
	const i32 nNodes = desc.numX*desc.numY*desc.numZ;
	const i32 k = desc.numCoeffs;
	const i32 h = params.fineRadius;
	const i32 fineDim = 2*h + 1;
	const i32 nFine = fineDim*fineDim*fineDim;

	for (i32 node = dbg::globalThreadIdx1d(); node < nNodes; node += dbg::globalThreadStride1d()) {
		const i32 ix = node % desc.numX;
		const i32 iy = (node / desc.numX) % desc.numY;
		const i32 iz = node / (desc.numX*desc.numY);

		const vec3 g{f32(ix), f32(iy), f32(iz)};
		const vec3 xCenter = gridToWorld(&desc, g);
		const vec3 centerQ = worldToQ(&desc, xCenter);

		f32 local[ADSDF_MAX_COEFFS];
		f32 global[ADSDF_MAX_COEFFS];
		zeroCoeffs(local);

		i32 fineIdx = 0;
		for (i32 oz = -h; oz <= h; ++oz) {
			for (i32 oy = -h; oy <= h; ++oy) {
				for (i32 ox = -h; ox <= h; ++ox) {
					const vec3 dq = fineOffsetQ(&desc, &params, ox, oy, oz);
					const vec3 sampleWorld = qToWorld(&desc, centerQ + dq);
					const f32 value = evalProceduralSdf(&source, sampleWorld);

					for (i32 coeffIdx = 0; coeffIdx < k; ++coeffIdx) {
						local[coeffIdx] += pinv_d[size_t(coeffIdx)*size_t(nFine) + size_t(fineIdx)]*value;
					}

					++fineIdx;
				}
			}
		}

		translateLocalToGlobal(global, local, desc.degree, centerQ);

		for (i32 groupIdx = 0; groupIdx < desc.numCoeffGroups; ++groupIdx) {
			grid.coeffGroups_d[coeffGroupIdx(&desc, groupIdx, ix, iy, iz)] =
				coeffGroupFromArray(global, groupIdx, k);
		}
	}
}

/** Launch one-node-per-thread procedural LSQ coefficient construction. */
inline void launchBuildProceduralLsq(
	AdsdfLinearGrid *grid,
	const f32 *pinv_d,
	AdsdfLsqParams params,
	ProceduralSdf source,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(grid != nullptr);
	SPR_ASSERT(grid->coeffGroups_d != nullptr);
	SPR_ASSERT(pinv_d != nullptr);

	const i32 n = i32(numNodes(&grid->desc));
	const i32 threadsPerBlock = ADSDF_DEFAULT_THREADS_PER_BLOCK;
	const i32 numBlocks = dbg::numBlocks(n, threadsPerBlock);

	buildProceduralLsq<<<numBlocks, threadsPerBlock, 0, stream>>>(
		*grid,
		pinv_d,
		params,
		source);
	CUDA_LAUNCH_CHECK();
}

#endif

} // namespace adsdf
