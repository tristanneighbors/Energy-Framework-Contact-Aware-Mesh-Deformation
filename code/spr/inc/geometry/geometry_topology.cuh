#pragma once

/**
 * @file geometry_topology.cuh
 * @brief Basic triangle-mesh one-ring topology and vertex area utilities.
 */

#include "geometry_query.cuh"

#include <algorithm>
#include <vector>

namespace geom {

struct MeshEdge {
	vid v0;
	vid v1;
};

struct MeshTopology {
	std::vector<MeshEdge> edges;
	std::vector<i32> neighborOffsets;
	std::vector<vid> neighbors;
};

inline bool isValidTriIndex(MeshView mesh, MeshTri tri) {
	return
		tri.v0 >= 0 && tri.v0 < mesh.numPositions &&
		tri.v1 >= 0 && tri.v1 < mesh.numPositions &&
		tri.v2 >= 0 && tri.v2 < mesh.numPositions;
}

inline MeshEdge makeSortedEdge(vid a, vid b) {
	MeshEdge edge = {};
	if (a < b) {
		edge.v0 = a;
		edge.v1 = b;
	} else {
		edge.v0 = b;
		edge.v1 = a;
	}
	return edge;
}

inline void clearMeshTopology(MeshTopology *topology) {
	if (topology == nullptr) {
		return;
	}

	topology->edges.clear();
	topology->neighborOffsets.clear();
	topology->neighbors.clear();
}

inline void buildMeshTopology(MeshTopology *topologyOut, MeshView mesh) {
	if (topologyOut == nullptr) {
		return;
	}

	clearMeshTopology(topologyOut);
	if (mesh.numPositions <= 0 || mesh.numTriangles <= 0 || mesh.triangles == nullptr) {
		topologyOut->neighborOffsets.assign(size_t(mesh.numPositions) + 1, 0);
		return;
	}

	std::vector<MeshEdge> edges;
	edges.reserve(size_t(mesh.numTriangles)*3u);

	for (i32 triIdx = 0; triIdx < mesh.numTriangles; ++triIdx) {
		const MeshTri tri = mesh.triangles[triIdx];
		if (!isValidTriIndex(mesh, tri)) {
			continue;
		}

		if (tri.v0 != tri.v1) { edges.push_back(makeSortedEdge(tri.v0, tri.v1)); }
		if (tri.v1 != tri.v2) { edges.push_back(makeSortedEdge(tri.v1, tri.v2)); }
		if (tri.v2 != tri.v0) { edges.push_back(makeSortedEdge(tri.v2, tri.v0)); }
	}

	std::sort(
		edges.begin(),
		edges.end(),
		[](MeshEdge a, MeshEdge b) {
			return a.v0 < b.v0 || (a.v0 == b.v0 && a.v1 < b.v1);
		});

	edges.erase(
		std::unique(
			edges.begin(),
			edges.end(),
			[](MeshEdge a, MeshEdge b) {
				return a.v0 == b.v0 && a.v1 == b.v1;
			}),
		edges.end());

	topologyOut->edges = std::move(edges);

	std::vector<i32> degree(size_t(mesh.numPositions), 0);
	for (MeshEdge edge : topologyOut->edges) {
		++degree[size_t(edge.v0)];
		++degree[size_t(edge.v1)];
	}

	topologyOut->neighborOffsets.assign(size_t(mesh.numPositions) + 1, 0);
	for (i32 v = 0; v < mesh.numPositions; ++v) {
		topologyOut->neighborOffsets[size_t(v) + 1] =
			topologyOut->neighborOffsets[size_t(v)] + degree[size_t(v)];
	}

	topologyOut->neighbors.assign(size_t(topologyOut->neighborOffsets.back()), GEOM_INVALID_IDX);
	std::vector<i32> cursor = topologyOut->neighborOffsets;
	for (MeshEdge edge : topologyOut->edges) {
		topologyOut->neighbors[size_t(cursor[size_t(edge.v0)]++)] = edge.v1;
		topologyOut->neighbors[size_t(cursor[size_t(edge.v1)]++)] = edge.v0;
	}

	for (i32 v = 0; v < mesh.numPositions; ++v) {
		const i32 begin = topologyOut->neighborOffsets[size_t(v)];
		const i32 end = topologyOut->neighborOffsets[size_t(v) + 1];
		std::sort(
			topologyOut->neighbors.begin() + begin,
			topologyOut->neighbors.begin() + end);
	}
}

inline void buildMeshTopology(MeshTopology *topologyOut, const Mesh *mesh) {
	buildMeshTopology(topologyOut, viewMesh(mesh));
}

inline void computeVertexAreas(std::vector<f32> *areasOut, MeshView mesh) {
	if (areasOut == nullptr) {
		return;
	}

	areasOut->assign(size_t(mesh.numPositions), 0.0f);
	if (mesh.positions == nullptr || mesh.triangles == nullptr) {
		return;
	}

	for (i32 triIdx = 0; triIdx < mesh.numTriangles; ++triIdx) {
		const MeshTri tri = mesh.triangles[triIdx];
		if (!isValidTriIndex(mesh, tri)) {
			continue;
		}

		const vec3 a = mesh.positions[tri.v0];
		const vec3 b = mesh.positions[tri.v1];
		const vec3 c = mesh.positions[tri.v2];
		const f32 areaThird = triangleArea(a, b, c)/3.0f;
		(*areasOut)[size_t(tri.v0)] += areaThird;
		(*areasOut)[size_t(tri.v1)] += areaThird;
		(*areasOut)[size_t(tri.v2)] += areaThird;
	}
}

inline void computeVertexAreas(std::vector<f32> *areasOut, const Mesh *mesh) {
	computeVertexAreas(areasOut, viewMesh(mesh));
}

inline f32 totalArea(const std::vector<f32> &areas) {
	f32 total = 0.0f;
	for (f32 area : areas) {
		total += area;
	}
	return total;
}

inline void makeHarmonicVertexDisplacementSeed(
	std::vector<vec3> *seededPositionsOut,
	const std::vector<vec3> &restPositions,
	const MeshTopology &topology,
	const std::vector<vid> &fixedVertices,
	const std::vector<vec3> &fixedTargets,
	i32 iterations) {
	if (seededPositionsOut == nullptr) {
		return;
	}

	*seededPositionsOut = restPositions;
	const i32 n = i32(restPositions.size());
	if (n <= 0 || iterations <= 0 || topology.neighborOffsets.size() < size_t(n) + 1u) {
		return;
	}

	std::vector<char> isFixed(size_t(n), 0);
	std::vector<vec3> displacement(size_t(n), vec3(0.0f));
	std::vector<vec3> next = displacement;
	const size_t numFixed = std::min(fixedVertices.size(), fixedTargets.size());

	for (size_t i = 0; i < numFixed; ++i) {
		const vid vertexIdx = fixedVertices[i];
		if (vertexIdx < 0 || vertexIdx >= n) {
			continue;
		}

		const size_t idx = size_t(vertexIdx);
		isFixed[idx] = 1;
		displacement[idx] = fixedTargets[i] - restPositions[idx];
		next[idx] = displacement[idx];
	}

	for (i32 iter = 0; iter < iterations; ++iter) {
		for (i32 vertexIdx = 0; vertexIdx < n; ++vertexIdx) {
			const size_t idx = size_t(vertexIdx);
			if (isFixed[idx] != 0) {
				next[idx] = displacement[idx];
				continue;
			}

			const i32 begin = topology.neighborOffsets[idx];
			const i32 end = topology.neighborOffsets[idx + 1u];
			vec3 sum(0.0f);
			i32 count = 0;
			for (i32 k = begin; k < end; ++k) {
				const vid neighbor = topology.neighbors[size_t(k)];
				if (neighbor < 0 || neighbor >= n) {
					continue;
				}
				sum += displacement[size_t(neighbor)];
				++count;
			}

			next[idx] = count > 0 ? sum/f32(count) : displacement[idx];
		}
		displacement.swap(next);
	}

	for (i32 vertexIdx = 0; vertexIdx < n; ++vertexIdx) {
		(*seededPositionsOut)[size_t(vertexIdx)] =
			restPositions[size_t(vertexIdx)] + displacement[size_t(vertexIdx)];
	}
}

} // namespace geom
