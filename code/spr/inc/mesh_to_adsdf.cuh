#pragma once

/**
 * @file mesh_to_adsdf.cuh
 * @brief Header-only bridge from geom::Mesh/BVH signed-distance queries to ADSDF LSQ construction.
 *
 * This is glue code between the geometry and ADSDF modules. Neither module
 * includes this header; users include it explicitly when building an ADSDF from
 * mesh geometry.
 */

#include "adsdf.cuh"
#include "geometry.cuh"
#include "dbg/cuda_utils.cuh"

namespace adsdf {

/** Device/host view of a mesh signed-distance source. */
struct MeshSdfSource {
	geom::MeshView mesh;
	geom::BvhView bvh;

	geom::MeshSignMethod signMethod;
	vec3 parityRayDir;
	f32 parityRayTMin;
	f32 parityRayTMax;
	f32 nearSurfaceEps;
	i32 maxParityHits;
	bool shouldUseNormalSignNearSurface;
};

/** Parameters for building an ADSDF from a GPU mesh and GPU BVH. */
struct MeshAdsdfBuildParams {
	AdsdfLsqParams lsq;

	geom::MeshSignMethod signMethod;
	vec3 parityRayDir;
	f32 parityRayTMin;
	f32 parityRayTMax;
	f32 nearSurfaceEps;
	i32 maxParityHits;
	bool shouldUseNormalSignNearSurface;
};

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 normalSignedDistanceFromClosest(geom::ClosestPointResult closest, vec3 p) {
	const f32 dist = sprSqrt<f32>(closest.dist2);
	const f32 n2 = sqNorm(closest.normal);
	if (n2 <= 1.0e-12f || !sprIsFinite(n2)) {
		return dist;
	}

	const f32 side = dot(p - closest.point, closest.normal);
	return side < 0.0f ? -dist : dist;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 safeParityRayDir(vec3 dir) {
	const f32 d2 = sqNorm(dir);
	if (d2 <= 1.0e-12f || !sprIsFinite(d2)) {
		return geom::defaultSignRayDir();
	}
	return dir*sprRsqrt<f32>(d2);
}

inline MeshAdsdfBuildParams defaultMeshAdsdfBuildParams() {
	MeshAdsdfBuildParams params = {};
	params.lsq.fineRadius = 2;
	params.lsq.fineExtent = 0.6f;
	params.lsq.regularization = 1.0e-7f;
	params.signMethod = geom::MeshSignMethod::RayParity;
	params.parityRayDir = geom::defaultSignRayDir();
	params.parityRayTMin = 1.0e-4f;
	params.parityRayTMax = F32_MAX;
	params.nearSurfaceEps = 1.0e-4f;
	params.maxParityHits = GEOM_RAY_COUNT_MAX_HITS;
	params.shouldUseNormalSignNearSurface = true;
	return params;
}

inline MeshSdfSource makeMeshSdfSource(
	geom::MeshView mesh,
	geom::BvhView bvh,
	const MeshAdsdfBuildParams *paramsIn) {
	const MeshAdsdfBuildParams defaults = defaultMeshAdsdfBuildParams();
	const MeshAdsdfBuildParams *params = paramsIn != nullptr ? paramsIn : &defaults;

	MeshSdfSource source = {};
	source.mesh = mesh;
	source.bvh = bvh;
	source.signMethod = params->signMethod;
	source.parityRayDir = safeParityRayDir(params->parityRayDir);
	source.parityRayTMin = params->parityRayTMin;
	source.parityRayTMax = params->parityRayTMax;
	source.nearSurfaceEps = params->nearSurfaceEps;
	source.maxParityHits = params->maxParityHits;
	source.shouldUseNormalSignNearSurface = params->shouldUseNormalSignNearSurface;
	return source;
}

inline MeshSdfSource makeMeshSdfSource(
	const geom::GpuMesh *mesh,
	const geom::GpuBvh *bvh,
	const MeshAdsdfBuildParams *params) {
	return makeMeshSdfSource(
		geom::viewGpuMesh(mesh),
		geom::viewGpuBvh(bvh),
		params);
}

/** Query the mesh/BVH source as a signed distance field. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalMeshSdf(MeshSdfSource source, vec3 p) {
	geom::ClosestPointResult closest;
	if (!geom::closestPointBvh(&closest, source.mesh, source.bvh, p)) {
		return F32_MAX;
	}

	const f32 dist = sprSqrt<f32>(closest.dist2);
	if (source.signMethod == geom::MeshSignMethod::Unsigned) {
		return dist;
	}

	if (source.signMethod == geom::MeshSignMethod::NormalPseudo) {
		return normalSignedDistanceFromClosest(closest, p);
	}

	if (source.shouldUseNormalSignNearSurface && dist <= source.nearSurfaceEps) {
		return normalSignedDistanceFromClosest(closest, p);
	}

	const vec3 dir = safeParityRayDir(source.parityRayDir);
	const f32 tMin = source.parityRayTMin > 0.0f ? source.parityRayTMin : 1.0e-4f;
	const f32 tMax = source.parityRayTMax > tMin ? source.parityRayTMax : F32_MAX;
	const i32 maxHits = source.maxParityHits > 0 ? source.maxParityHits : GEOM_RAY_COUNT_MAX_HITS;
	const geom::Ray ray = geom::makeRay(p, dir, tMin, tMax);
	const i32 count = geom::countRayIntersectionsBvh(source.mesh, source.bvh, ray, maxHits);

	return (count & 1) ? -dist : dist;
}

static SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 meshFineOffsetQ(
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

#if defined(__CUDACC__)

static __global__
void buildMeshLsq(
	AdsdfLinearGrid grid,
	const f32 *pinv_d,
	AdsdfLsqParams lsqParams,
	MeshSdfSource source) {
	const AdsdfDesc desc = grid.desc;
	const i32 nNodes = desc.numX*desc.numY*desc.numZ;
	const i32 k = desc.numCoeffs;
	const i32 h = lsqParams.fineRadius;
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
					const vec3 dq = meshFineOffsetQ(&desc, &lsqParams, ox, oy, oz);
					const vec3 sampleWorld = qToWorld(&desc, centerQ + dq);
					const f32 value = evalMeshSdf(source, sampleWorld);

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

inline void launchBuildMeshLsq(
	AdsdfLinearGrid *grid,
	const f32 *pinv_d,
	MeshAdsdfBuildParams params,
	MeshSdfSource source,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(grid != nullptr);
	SPR_ASSERT(grid->coeffGroups_d != nullptr);
	SPR_ASSERT(pinv_d != nullptr);

	const i32 n = i32(numNodes(&grid->desc));
	const i32 threadsPerBlock = ADSDF_DEFAULT_THREADS_PER_BLOCK;
	const i32 numBlocks = dbg::numBlocks(n, threadsPerBlock);

	buildMeshLsq<<<numBlocks, threadsPerBlock, 0, stream>>>(
		*grid,
		pinv_d,
		params.lsq,
		source);
	CUDA_LAUNCH_CHECK();
}

inline void launchBuildMeshLsq(
	AdsdfLinearGrid *grid,
	const f32 *pinv_d,
	MeshAdsdfBuildParams params,
	const geom::GpuMesh *mesh,
	const geom::GpuBvh *bvh,
	cudaStream_t stream = nullptr) {
	const MeshSdfSource source = makeMeshSdfSource(mesh, bvh, &params);
	launchBuildMeshLsq(grid, pinv_d, params, source, stream);
}

/** Convenience path that allocates the LSQ pseudoinverse, launches, synchronizes, and frees it. */
inline void buildMeshLsqBlocking(
	AdsdfLinearGrid *grid,
	const geom::GpuMesh *mesh,
	const geom::GpuBvh *bvh,
	MeshAdsdfBuildParams params,
	cudaStream_t stream = nullptr) {
	SPR_ASSERT(grid != nullptr);
	SPR_ASSERT(mesh != nullptr);
	SPR_ASSERT(bvh != nullptr);

	f32 *pinv_d = nullptr;
	allocAndUploadLsqPseudoinverse(&pinv_d, &grid->desc, &params.lsq);
	SPR_ASSERT(pinv_d != nullptr);
	if (pinv_d == nullptr) {
		return;
	}

	launchBuildMeshLsq(grid, pinv_d, params, mesh, bvh, stream);
	CUDA_CHECK(cudaStreamSynchronize(stream));
	freeLsqPseudoinverse(&pinv_d);
}

#endif

} // namespace adsdf
