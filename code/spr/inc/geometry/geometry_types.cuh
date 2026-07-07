#pragma once

/**
 * @file geometry_types.cuh
 * @brief POD-like mesh, ray, AABB, hit, and BVH data types.
 */

#include "spr_global_include.h"
#include "spr_math.cuh"

#include <vector>

#ifndef GEOM_INVALID_IDX
#define GEOM_INVALID_IDX (-1)
#endif

#ifndef GEOM_BVH_STACK_SIZE
#define GEOM_BVH_STACK_SIZE 96
#endif

#ifndef GEOM_RAY_COUNT_MAX_HITS
#define GEOM_RAY_COUNT_MAX_HITS 4096
#endif

namespace geom {

using vid = i32;
using fid = i32;

/** Sign policy for mesh distance queries. */
enum class MeshSignMethod : i32 {
	Unsigned,
	RayParity,
	NormalPseudo,
};

/** Triangle with independent OBJ position/texcoord/normal indices. */
struct MeshTri {
	vid v0;
	vid v1;
	vid v2;

	i32 vt0;
	i32 vt1;
	i32 vt2;

	i32 vn0;
	i32 vn1;
	i32 vn2;
};

/** Host-side mesh storage. Positions and triangles are required; other arrays are optional. */
struct Mesh {
	std::vector<vec3> positions;
	std::vector<vec2> texcoords;
	std::vector<vec3> normals;
	std::vector<MeshTri> triangles;

	// Derived convenience data. `computeTriangleNormals` and `computeVertexNormals` fill these.
	std::vector<vec3> triNormals;
	std::vector<vec3> vertexNormals;
};

/** Non-owning mesh view usable in host and device query code. */
struct MeshView {
	const vec3 *positions;
	const vec2 *texcoords;
	const vec3 *normals;
	const MeshTri *triangles;
	const vec3 *triNormals;
	const vec3 *vertexNormals;

	i32 numPositions;
	i32 numTexcoords;
	i32 numNormals;
	i32 numTriangles;
};

/** Owning GPU mirror of a Mesh. */
struct GpuMesh {
	vec3 *positions_d;
	vec2 *texcoords_d;
	vec3 *normals_d;
	MeshTri *triangles_d;
	vec3 *triNormals_d;
	vec3 *vertexNormals_d;

	i32 numPositions;
	i32 numTexcoords;
	i32 numNormals;
	i32 numTriangles;

	bool ownsMemory;
};

struct Ray {
	vec3 origin;
	vec3 dir;
	f32 tMin;
	f32 tMax;
};

struct Aabb {
	vec3 lower;
	vec3 upper;
};

struct RayTriHit {
	f32 t;
	f32 u;
	f32 v;
	f32 w;
	fid triIdx;
};

struct RayHit {
	f32 t;
	vec3 position;
	vec3 normal;
	f32 u;
	f32 v;
	f32 w;
	fid triIdx;
};

struct ClosestPointResult {
	f32 dist2;
	vec3 point;
	vec3 normal;
	f32 u;
	f32 v;
	f32 w;
	fid triIdx;
};

/** Compact binary BVH node. Leaves have `numTris > 0`. */
struct BvhNode {
	Aabb bounds;

	i32 left;
	i32 right;

	i32 firstTri;
	i32 numTris;
};

/** Host-side BVH storage. `triIndices` maps leaf ranges to mesh triangle indices. */
struct Bvh {
	std::vector<BvhNode> nodes;
	std::vector<i32> triIndices;
	Aabb bounds;
	i32 root;
	i32 leafSize;
};

/** Owning GPU mirror of a Bvh. */
struct GpuBvh {
	BvhNode *nodes_d;
	i32 *triIndices_d;
	i32 numNodes;
	i32 numTriIndices;
	i32 root;
	i32 leafSize;
	Aabb bounds;
	bool ownsMemory;
};

/** Non-owning BVH view usable in host and device query code. */
struct BvhView {
	const BvhNode *nodes;
	const i32 *triIndices;
	i32 numNodes;
	i32 numTriIndices;
	i32 root;
	Aabb bounds;
};

struct BvhBuildConfig {
	i32 leafSize;
	i32 maxDepth;
	bool shouldUseMedianSplit;
};

struct ObjLoadOptions {
	bool shouldTriangulate;
	bool shouldComputeMissingNormals;
	bool shouldCenter;
	bool shouldNormalizeToUnitBox;
	bool shouldPrintWarnings;
};

inline BvhBuildConfig defaultBvhBuildConfig() {
	BvhBuildConfig config = {};
	config.leafSize = 4;
	config.maxDepth = 64;
	config.shouldUseMedianSplit = true;
	return config;
}

inline ObjLoadOptions defaultObjLoadOptions() {
	ObjLoadOptions options = {};
	options.shouldTriangulate = true;
	options.shouldComputeMissingNormals = true;
	options.shouldCenter = false;
	options.shouldNormalizeToUnitBox = false;
	options.shouldPrintWarnings = true;
	return options;
}

inline MeshView viewMesh(const Mesh *mesh) {
	MeshView view = {};
	if (mesh == nullptr) {
		return view;
	}

	view.positions = mesh->positions.empty() ? nullptr : mesh->positions.data();
	view.texcoords = mesh->texcoords.empty() ? nullptr : mesh->texcoords.data();
	view.normals = mesh->normals.empty() ? nullptr : mesh->normals.data();
	view.triangles = mesh->triangles.empty() ? nullptr : mesh->triangles.data();
	view.triNormals = mesh->triNormals.empty() ? nullptr : mesh->triNormals.data();
	view.vertexNormals = mesh->vertexNormals.empty() ? nullptr : mesh->vertexNormals.data();

	view.numPositions = static_cast<i32>(mesh->positions.size());
	view.numTexcoords = static_cast<i32>(mesh->texcoords.size());
	view.numNormals = static_cast<i32>(mesh->normals.size());
	view.numTriangles = static_cast<i32>(mesh->triangles.size());
	return view;
}

inline MeshView viewGpuMesh(const GpuMesh *mesh) {
	MeshView view = {};
	if (mesh == nullptr) {
		return view;
	}

	view.positions = mesh->positions_d;
	view.texcoords = mesh->texcoords_d;
	view.normals = mesh->normals_d;
	view.triangles = mesh->triangles_d;
	view.triNormals = mesh->triNormals_d;
	view.vertexNormals = mesh->vertexNormals_d;
	view.numPositions = mesh->numPositions;
	view.numTexcoords = mesh->numTexcoords;
	view.numNormals = mesh->numNormals;
	view.numTriangles = mesh->numTriangles;
	return view;
}

inline BvhView viewBvh(const Bvh *bvh) {
	BvhView view = {};
	if (bvh == nullptr) {
		view.root = GEOM_INVALID_IDX;
		return view;
	}

	view.nodes = bvh->nodes.empty() ? nullptr : bvh->nodes.data();
	view.triIndices = bvh->triIndices.empty() ? nullptr : bvh->triIndices.data();
	view.numNodes = static_cast<i32>(bvh->nodes.size());
	view.numTriIndices = static_cast<i32>(bvh->triIndices.size());
	view.root = bvh->root;
	view.bounds = bvh->bounds;
	return view;
}

inline BvhView viewGpuBvh(const GpuBvh *bvh) {
	BvhView view = {};
	if (bvh == nullptr) {
		view.root = GEOM_INVALID_IDX;
		return view;
	}

	view.nodes = bvh->nodes_d;
	view.triIndices = bvh->triIndices_d;
	view.numNodes = bvh->numNodes;
	view.numTriIndices = bvh->numTriIndices;
	view.root = bvh->root;
	view.bounds = bvh->bounds;
	return view;
}

} // namespace geom
