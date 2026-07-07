#pragma once

/**
 * @file geometry_bvh.cuh
 * @brief CPU-built, GPU-resident BVH and device traversal routines.
 */

#include "geometry_mesh.cuh"

#include <algorithm>

namespace geom {

void clearBvh(Bvh *bvh);
void buildBvh(Bvh *bvhOut, MeshView mesh, BvhBuildConfig config);
void buildBvh(Bvh *bvhOut, const Mesh *mesh, BvhBuildConfig config);
void buildBvh(Bvh *bvhOut, const Mesh *mesh);

void initGpuBvh(GpuBvh *bvhOut);
void freeGpuBvh(GpuBvh *bvh);
void uploadGpuBvh(GpuBvh *bvhOut, const Bvh *bvh);

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool isLeaf(const BvhNode *node) {
	return node->numTris > 0;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void unpackTriangle(vec3 *aOut, vec3 *bOut, vec3 *cOut, MeshView mesh, fid triIdx) {
	const MeshTri tri = mesh.triangles[triIdx];
	*aOut = mesh.positions[tri.v0];
	*bOut = mesh.positions[tri.v1];
	*cOut = mesh.positions[tri.v2];
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 meshTriangleNormal(MeshView mesh, fid triIdx) {
	if (mesh.triNormals != nullptr && triIdx >= 0 && triIdx < mesh.numTriangles) {
		return mesh.triNormals[triIdx];
	}

	vec3 a;
	vec3 b;
	vec3 c;
	unpackTriangle(&a, &b, &c, mesh, triIdx);
	return triangleNormal(a, b, c);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
ClosestPointResult makeInvalidClosestPointResult() {
	ClosestPointResult out;
	out.dist2 = F32_MAX;
	out.point = vec3(0.0f);
	out.normal = vec3(0.0f);
	out.u = 0.0f;
	out.v = 0.0f;
	out.w = 0.0f;
	out.triIdx = GEOM_INVALID_IDX;
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
RayHit makeInvalidRayHit() {
	RayHit out;
	out.t = F32_MAX;
	out.position = vec3(0.0f);
	out.normal = vec3(0.0f);
	out.u = 0.0f;
	out.v = 0.0f;
	out.w = 0.0f;
	out.triIdx = GEOM_INVALID_IDX;
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool closestPointBrute(ClosestPointResult *closestOut, MeshView mesh, vec3 p) {
	ClosestPointResult best = makeInvalidClosestPointResult();

	for (i32 triIdx = 0; triIdx < mesh.numTriangles; ++triIdx) {
		vec3 a;
		vec3 b;
		vec3 c;
		unpackTriangle(&a, &b, &c, mesh, triIdx);
		ClosestPointResult candidate = closestPointOnTriangle(p, a, b, c, triIdx);
		candidate.normal = meshTriangleNormal(mesh, triIdx);
		if (candidate.dist2 < best.dist2) {
			best = candidate;
		}
	}

	if (closestOut != nullptr) {
		*closestOut = best;
	}
	return best.triIdx != GEOM_INVALID_IDX;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool raycastBrute(RayHit *hitOut, MeshView mesh, Ray ray) {
	RayHit best = makeInvalidRayHit();
	Ray localRay = ray;

	for (i32 triIdx = 0; triIdx < mesh.numTriangles; ++triIdx) {
		vec3 a;
		vec3 b;
		vec3 c;
		unpackTriangle(&a, &b, &c, mesh, triIdx);

		RayTriHit triHit;
		if (intersectRayTriangle(&triHit, localRay, a, b, c, triIdx)) {
			localRay.tMax = triHit.t;
			best.t = triHit.t;
			best.u = triHit.u;
			best.v = triHit.v;
			best.w = triHit.w;
			best.triIdx = triIdx;
			best.position = rayPoint(ray, triHit.t);
			best.normal = meshTriangleNormal(mesh, triIdx);
		}
	}

	if (hitOut != nullptr) {
		*hitOut = best;
	}
	return best.triIdx != GEOM_INVALID_IDX;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool closestPointBvh(ClosestPointResult *closestOut, MeshView mesh, BvhView bvh, vec3 p) {
	if (bvh.nodes == nullptr || bvh.triIndices == nullptr || bvh.root < 0) {
		return closestPointBrute(closestOut, mesh, p);
	}

	ClosestPointResult best = makeInvalidClosestPointResult();
	i32 stack[GEOM_BVH_STACK_SIZE];
	i32 stackSize = 0;
	stack[stackSize++] = bvh.root;

	while (stackSize > 0) {
		const i32 nodeIdx = stack[--stackSize];
		if (nodeIdx < 0 || nodeIdx >= bvh.numNodes) {
			continue;
		}

		const BvhNode node = bvh.nodes[nodeIdx];
		const f32 nodeDist2 = sqDistancePointAabb(p, node.bounds);
		if (nodeDist2 > best.dist2) {
			continue;
		}

		if (isLeaf(&node)) {
			for (i32 i = 0; i < node.numTris; ++i) {
				const i32 indexIdx = node.firstTri + i;
				if (indexIdx < 0 || indexIdx >= bvh.numTriIndices) {
					continue;
				}

				const fid triIdx = bvh.triIndices[indexIdx];
				vec3 a;
				vec3 b;
				vec3 c;
				unpackTriangle(&a, &b, &c, mesh, triIdx);

				ClosestPointResult candidate = closestPointOnTriangle(p, a, b, c, triIdx);
				candidate.normal = meshTriangleNormal(mesh, triIdx);
				if (candidate.dist2 < best.dist2) {
					best = candidate;
				}
			}
			continue;
		}

		const i32 leftIdx = node.left;
		const i32 rightIdx = node.right;
		f32 leftDist2 = F32_MAX;
		f32 rightDist2 = F32_MAX;

		if (leftIdx >= 0 && leftIdx < bvh.numNodes) {
			leftDist2 = sqDistancePointAabb(p, bvh.nodes[leftIdx].bounds);
		}
		if (rightIdx >= 0 && rightIdx < bvh.numNodes) {
			rightDist2 = sqDistancePointAabb(p, bvh.nodes[rightIdx].bounds);
		}

		const i32 nearIdx = leftDist2 <= rightDist2 ? leftIdx : rightIdx;
		const i32 farIdx = leftDist2 <= rightDist2 ? rightIdx : leftIdx;
		const f32 nearDist2 = leftDist2 <= rightDist2 ? leftDist2 : rightDist2;
		const f32 farDist2 = leftDist2 <= rightDist2 ? rightDist2 : leftDist2;

		if (farIdx >= 0 && farDist2 <= best.dist2 && stackSize < GEOM_BVH_STACK_SIZE) {
			stack[stackSize++] = farIdx;
		}
		if (nearIdx >= 0 && nearDist2 <= best.dist2 && stackSize < GEOM_BVH_STACK_SIZE) {
			stack[stackSize++] = nearIdx;
		}
	}

	if (closestOut != nullptr) {
		*closestOut = best;
	}
	return best.triIdx != GEOM_INVALID_IDX;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
bool raycastBvh(RayHit *hitOut, MeshView mesh, BvhView bvh, Ray ray) {
	if (bvh.nodes == nullptr || bvh.triIndices == nullptr || bvh.root < 0) {
		return raycastBrute(hitOut, mesh, ray);
	}

	RayHit best = makeInvalidRayHit();
	Ray localRay = ray;
	i32 stack[GEOM_BVH_STACK_SIZE];
	f32 nearStack[GEOM_BVH_STACK_SIZE];
	i32 stackSize = 0;

	f32 rootNear;
	f32 rootFar;
	if (!intersectRayAabb(&rootNear, &rootFar, localRay, bvh.bounds)) {
		if (hitOut != nullptr) {
			*hitOut = best;
		}
		return false;
	}

	stack[stackSize] = bvh.root;
	nearStack[stackSize] = rootNear;
	++stackSize;

	while (stackSize > 0) {
		--stackSize;
		const i32 nodeIdx = stack[stackSize];
		const f32 nodeNear = nearStack[stackSize];
		if (nodeNear > localRay.tMax || nodeIdx < 0 || nodeIdx >= bvh.numNodes) {
			continue;
		}

		const BvhNode node = bvh.nodes[nodeIdx];
		if (isLeaf(&node)) {
			for (i32 i = 0; i < node.numTris; ++i) {
				const i32 indexIdx = node.firstTri + i;
				if (indexIdx < 0 || indexIdx >= bvh.numTriIndices) {
					continue;
				}

				const fid triIdx = bvh.triIndices[indexIdx];
				vec3 a;
				vec3 b;
				vec3 c;
				unpackTriangle(&a, &b, &c, mesh, triIdx);

				RayTriHit triHit;
				if (intersectRayTriangle(&triHit, localRay, a, b, c, triIdx)) {
					localRay.tMax = triHit.t;
					best.t = triHit.t;
					best.u = triHit.u;
					best.v = triHit.v;
					best.w = triHit.w;
					best.triIdx = triIdx;
					best.position = rayPoint(ray, triHit.t);
					best.normal = meshTriangleNormal(mesh, triIdx);
				}
			}
			continue;
		}

		const i32 leftIdx = node.left;
		const i32 rightIdx = node.right;
		f32 leftNear = F32_MAX;
		f32 rightNear = F32_MAX;
		f32 tmpFar = F32_MAX;
		bool hasLeft = false;
		bool hasRight = false;

		if (leftIdx >= 0 && leftIdx < bvh.numNodes) {
			hasLeft = intersectRayAabb(&leftNear, &tmpFar, localRay, bvh.nodes[leftIdx].bounds);
		}
		if (rightIdx >= 0 && rightIdx < bvh.numNodes) {
			hasRight = intersectRayAabb(&rightNear, &tmpFar, localRay, bvh.nodes[rightIdx].bounds);
		}

		const bool leftNearer = leftNear <= rightNear;
		const i32 nearIdx = leftNearer ? leftIdx : rightIdx;
		const i32 farIdx = leftNearer ? rightIdx : leftIdx;
		const f32 nearT = leftNearer ? leftNear : rightNear;
		const f32 farT = leftNearer ? rightNear : leftNear;
		const bool hasNear = leftNearer ? hasLeft : hasRight;
		const bool hasFar = leftNearer ? hasRight : hasLeft;

		if (hasFar && farT <= localRay.tMax && stackSize < GEOM_BVH_STACK_SIZE) {
			stack[stackSize] = farIdx;
			nearStack[stackSize] = farT;
			++stackSize;
		}
		if (hasNear && nearT <= localRay.tMax && stackSize < GEOM_BVH_STACK_SIZE) {
			stack[stackSize] = nearIdx;
			nearStack[stackSize] = nearT;
			++stackSize;
		}
	}

	if (hitOut != nullptr) {
		*hitOut = best;
	}
	return best.triIdx != GEOM_INVALID_IDX;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
i32 countRayIntersectionsBvh(MeshView mesh, BvhView bvh, Ray ray, i32 maxHits) {
	Ray localRay = ray;
	i32 count = 0;
	const i32 hitLimit = maxHits > 0 ? maxHits : GEOM_RAY_COUNT_MAX_HITS;

	for (i32 iter = 0; iter < hitLimit; ++iter) {
		RayHit hit;
		if (!raycastBvh(&hit, mesh, bvh, localRay)) {
			break;
		}

		++count;
		localRay.tMin = hit.t + 1.0e-4f;
		if (localRay.tMin >= localRay.tMax) {
			break;
		}
	}

	return count;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 unsignedDistanceBvh(MeshView mesh, BvhView bvh, vec3 p) {
	ClosestPointResult closest;
	if (!closestPointBvh(&closest, mesh, bvh, p)) {
		return F32_MAX;
	}
	return sprSqrt<f32>(closest.dist2);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 signedDistanceBvh(MeshView mesh, BvhView bvh, vec3 p, MeshSignMethod signMethod) {
	ClosestPointResult closest;
	if (!closestPointBvh(&closest, mesh, bvh, p)) {
		return F32_MAX;
	}

	f32 distance = sprSqrt<f32>(closest.dist2);
	if (signMethod == MeshSignMethod::Unsigned) {
		return distance;
	}

	if (signMethod == MeshSignMethod::NormalPseudo) {
		const f32 side = dot(p - closest.point, closest.normal);
		return side < 0.0f ? -distance : distance;
	}

	const vec3 dir = defaultSignRayDir();
	const Ray ray = makeRay(p, dir, 1.0e-4f, F32_MAX);
	const i32 count = countRayIntersectionsBvh(mesh, bvh, ray, GEOM_RAY_COUNT_MAX_HITS);
	return (count & 1) ? -distance : distance;
}

struct BvhTriInfo {
	Aabb bounds;
	vec3 centroid;
};

struct BvhBuildScratch {
	Bvh *bvh;
	MeshView mesh;
	std::vector<BvhTriInfo> triInfo;
	BvhBuildConfig config;
};

static inline Aabb computeTriBounds(MeshView mesh, fid triIdx) {
	vec3 a;
	vec3 b;
	vec3 c;
	unpackTriangle(&a, &b, &c, mesh, triIdx);
	return triangleAabb(a, b, c);
}

static inline i32 buildBvhNodeRecursive(BvhBuildScratch *scratch, i32 first, i32 count, i32 depth) {
	const i32 nodeIdx = static_cast<i32>(scratch->bvh->nodes.size());
	scratch->bvh->nodes.push_back(BvhNode{});

	Aabb nodeBounds = emptyAabb();
	Aabb centroidBounds = emptyAabb();
	for (i32 i = 0; i < count; ++i) {
		const fid triIdx = scratch->bvh->triIndices[first + i];
		growAabb(&nodeBounds, scratch->triInfo[triIdx].bounds);
		growAabbPoint(&centroidBounds, scratch->triInfo[triIdx].centroid);
	}

	const i32 leafSize = scratch->config.leafSize > 0 ? scratch->config.leafSize : 4;
	const i32 maxDepth = scratch->config.maxDepth > 0 ? scratch->config.maxDepth : 64;
	const bool shouldMakeLeaf = count <= leafSize || depth >= maxDepth;

	if (shouldMakeLeaf) {
		BvhNode node = {};
		node.bounds = nodeBounds;
		node.left = GEOM_INVALID_IDX;
		node.right = GEOM_INVALID_IDX;
		node.firstTri = first;
		node.numTris = count;
		scratch->bvh->nodes[nodeIdx] = node;
		return nodeIdx;
	}

	const i32 axis = largestAxis(aabbExtent(centroidBounds));
	const i32 mid = first + count/2;

	if (scratch->config.shouldUseMedianSplit) {
		std::nth_element(
			scratch->bvh->triIndices.begin() + first,
			scratch->bvh->triIndices.begin() + mid,
			scratch->bvh->triIndices.begin() + first + count,
			[&](i32 a, i32 b) {
				const f32 ca = component(scratch->triInfo[a].centroid, axis);
				const f32 cb = component(scratch->triInfo[b].centroid, axis);
				if (ca == cb) {
					return a < b;
				}
				return ca < cb;
			});
	} else {
		std::sort(
			scratch->bvh->triIndices.begin() + first,
			scratch->bvh->triIndices.begin() + first + count,
			[&](i32 a, i32 b) {
				const f32 ca = component(scratch->triInfo[a].centroid, axis);
				const f32 cb = component(scratch->triInfo[b].centroid, axis);
				if (ca == cb) {
					return a < b;
				}
				return ca < cb;
			});
	}

	const i32 left = buildBvhNodeRecursive(scratch, first, mid - first, depth + 1);
	const i32 right = buildBvhNodeRecursive(scratch, mid, first + count - mid, depth + 1);

	BvhNode node = {};
	node.bounds = nodeBounds;
	node.left = left;
	node.right = right;
	node.firstTri = GEOM_INVALID_IDX;
	node.numTris = 0;
	scratch->bvh->nodes[nodeIdx] = node;
	return nodeIdx;
}

inline void clearBvh(Bvh *bvh) {
	if (bvh == nullptr) {
		return;
	}

	bvh->nodes.clear();
	bvh->triIndices.clear();
	bvh->bounds = emptyAabb();
	bvh->root = GEOM_INVALID_IDX;
	bvh->leafSize = 0;
}

inline void buildBvh(Bvh *bvhOut, MeshView mesh, BvhBuildConfig config) {
	if (bvhOut == nullptr) {
		return;
	}

	clearBvh(bvhOut);
	bvhOut->leafSize = config.leafSize;

	if (mesh.numTriangles <= 0 || mesh.positions == nullptr || mesh.triangles == nullptr) {
		return;
	}

	bvhOut->triIndices.resize(mesh.numTriangles);
	for (i32 triIdx = 0; triIdx < mesh.numTriangles; ++triIdx) {
		bvhOut->triIndices[triIdx] = triIdx;
	}

	BvhBuildScratch scratch = {};
	scratch.bvh = bvhOut;
	scratch.mesh = mesh;
	scratch.config = config;
	scratch.triInfo.resize(mesh.numTriangles);

	for (i32 triIdx = 0; triIdx < mesh.numTriangles; ++triIdx) {
		const Aabb bounds = computeTriBounds(mesh, triIdx);
		scratch.triInfo[triIdx].bounds = bounds;
		scratch.triInfo[triIdx].centroid = aabbCentroid(bounds);
		growAabb(&bvhOut->bounds, bounds);
	}

	bvhOut->root = buildBvhNodeRecursive(&scratch, 0, mesh.numTriangles, 0);
}

inline void buildBvh(Bvh *bvhOut, const Mesh *mesh, BvhBuildConfig config) {
	buildBvh(bvhOut, viewMesh(mesh), config);
}

inline void buildBvh(Bvh *bvhOut, const Mesh *mesh) {
	buildBvh(bvhOut, mesh, defaultBvhBuildConfig());
}

inline void initGpuBvh(GpuBvh *bvhOut) {
	if (bvhOut == nullptr) {
		return;
	}

	bvhOut->nodes_d = nullptr;
	bvhOut->triIndices_d = nullptr;
	bvhOut->numNodes = 0;
	bvhOut->numTriIndices = 0;
	bvhOut->root = GEOM_INVALID_IDX;
	bvhOut->leafSize = 0;
	bvhOut->bounds = emptyAabb();
	bvhOut->ownsMemory = false;
}

inline void freeGpuBvh(GpuBvh *bvh) {
	if (bvh == nullptr) {
		return;
	}

	if (bvh->ownsMemory) {
		if (bvh->nodes_d != nullptr) { CUDA_CHECK(cudaFree(bvh->nodes_d)); }
		if (bvh->triIndices_d != nullptr) { CUDA_CHECK(cudaFree(bvh->triIndices_d)); }
	}

	initGpuBvh(bvh);
}

inline void uploadGpuBvh(GpuBvh *bvhOut, const Bvh *bvh) {
	if (bvhOut == nullptr || bvh == nullptr) {
		return;
	}

	freeGpuBvh(bvhOut);
	uploadVector(&bvhOut->nodes_d, bvh->nodes);
	uploadVector(&bvhOut->triIndices_d, bvh->triIndices);

	bvhOut->numNodes = static_cast<i32>(bvh->nodes.size());
	bvhOut->numTriIndices = static_cast<i32>(bvh->triIndices.size());
	bvhOut->root = bvh->root;
	bvhOut->leafSize = bvh->leafSize;
	bvhOut->bounds = bvh->bounds;
	bvhOut->ownsMemory = true;
}

} // namespace geom
