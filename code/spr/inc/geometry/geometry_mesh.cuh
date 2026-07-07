#pragma once

/**
 * @file geometry_mesh.cuh
 * @brief Host mesh utility functions and GPU mesh upload helpers.
 */

#include "geometry_query.cuh"

#include "dbg/cuda_utils.cuh"

#include <cstdio>
#include <vector>

namespace geom {

inline void clearMesh(Mesh *mesh) {
	if (mesh == nullptr) {
		return;
	}

	mesh->positions.clear();
	mesh->texcoords.clear();
	mesh->normals.clear();
	mesh->triangles.clear();
	mesh->triNormals.clear();
	mesh->vertexNormals.clear();
}

inline void appendTriangle(Mesh *mesh, vid v0, vid v1, vid v2) {
	if (mesh == nullptr) {
		return;
	}

	MeshTri tri = {};
	tri.v0 = v0;
	tri.v1 = v1;
	tri.v2 = v2;
	tri.vt0 = GEOM_INVALID_IDX;
	tri.vt1 = GEOM_INVALID_IDX;
	tri.vt2 = GEOM_INVALID_IDX;
	tri.vn0 = GEOM_INVALID_IDX;
	tri.vn1 = GEOM_INVALID_IDX;
	tri.vn2 = GEOM_INVALID_IDX;
	mesh->triangles.push_back(tri);
}

inline void appendTriangle(Mesh *mesh, MeshTri tri) {
	if (mesh == nullptr) {
		return;
	}
	mesh->triangles.push_back(tri);
}

inline Aabb computeMeshBounds(MeshView mesh) {
	Aabb bounds = emptyAabb();
	for (i32 i = 0; i < mesh.numPositions; ++i) {
		growAabbPoint(&bounds, mesh.positions[i]);
	}
	return bounds;
}

inline Aabb computeMeshBounds(const Mesh *mesh) {
	return computeMeshBounds(viewMesh(mesh));
}

inline void computeTriangleNormals(Mesh *mesh) {
	if (mesh == nullptr) {
		return;
	}

	mesh->triNormals.resize(mesh->triangles.size());
	for (size_t triIdx = 0; triIdx < mesh->triangles.size(); ++triIdx) {
		const MeshTri tri = mesh->triangles[triIdx];
		if (
			tri.v0 < 0 || tri.v0 >= static_cast<i32>(mesh->positions.size()) ||
			tri.v1 < 0 || tri.v1 >= static_cast<i32>(mesh->positions.size()) ||
			tri.v2 < 0 || tri.v2 >= static_cast<i32>(mesh->positions.size())) {
			mesh->triNormals[triIdx] = vec3(0.0f);
			continue;
		}

		mesh->triNormals[triIdx] = triangleNormal(
			mesh->positions[tri.v0],
			mesh->positions[tri.v1],
			mesh->positions[tri.v2]);
	}
}

inline void computeVertexNormals(Mesh *mesh) {
	if (mesh == nullptr) {
		return;
	}

	mesh->vertexNormals.assign(mesh->positions.size(), vec3(0.0f));
	for (size_t triIdx = 0; triIdx < mesh->triangles.size(); ++triIdx) {
		const MeshTri tri = mesh->triangles[triIdx];
		if (
			tri.v0 < 0 || tri.v0 >= static_cast<i32>(mesh->positions.size()) ||
			tri.v1 < 0 || tri.v1 >= static_cast<i32>(mesh->positions.size()) ||
			tri.v2 < 0 || tri.v2 >= static_cast<i32>(mesh->positions.size())) {
			continue;
		}

		const vec3 a = mesh->positions[tri.v0];
		const vec3 b = mesh->positions[tri.v1];
		const vec3 c = mesh->positions[tri.v2];
		const vec3 areaNormal = cross(b - a, c - a);
		mesh->vertexNormals[tri.v0] += areaNormal;
		mesh->vertexNormals[tri.v1] += areaNormal;
		mesh->vertexNormals[tri.v2] += areaNormal;
	}

	for (vec3 &n : mesh->vertexNormals) {
		n = normalFromAreaVector(n);
	}
}

inline void ensureMeshNormals(Mesh *mesh) {
	if (mesh == nullptr) {
		return;
	}

	if (mesh->triNormals.size() != mesh->triangles.size()) {
		computeTriangleNormals(mesh);
	}
	if (mesh->vertexNormals.size() != mesh->positions.size()) {
		computeVertexNormals(mesh);
	}
}

inline bool validateMesh(const Mesh *mesh, bool shouldPrintWarnings = true) {
	if (mesh == nullptr) {
		if (shouldPrintWarnings) {
			std::fprintf(stderr, "geom: null mesh\n");
		}
		return false;
	}

	bool isValid = true;
	if (mesh->positions.empty()) {
		if (shouldPrintWarnings) {
			std::fprintf(stderr, "geom: mesh has no positions\n");
		}
		isValid = false;
	}

	for (size_t triIdx = 0; triIdx < mesh->triangles.size(); ++triIdx) {
		const MeshTri tri = mesh->triangles[triIdx];
		const bool triValid =
			tri.v0 >= 0 && tri.v0 < static_cast<i32>(mesh->positions.size()) &&
			tri.v1 >= 0 && tri.v1 < static_cast<i32>(mesh->positions.size()) &&
			tri.v2 >= 0 && tri.v2 < static_cast<i32>(mesh->positions.size());
		if (!triValid) {
			if (shouldPrintWarnings) {
				std::fprintf(stderr, "geom: triangle %zu has invalid position indices\n", triIdx);
			}
			isValid = false;
		}
	}

	return isValid;
}

template<class T>
static void uploadVector(T **dataOut_d, const std::vector<T> &src) {
	*dataOut_d = nullptr;
	if (src.empty()) {
		return;
	}

	const size_t size_bytes = src.size()*sizeof(T);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(dataOut_d), size_bytes));
	CUDA_CHECK(cudaMemcpy(*dataOut_d, src.data(), size_bytes, cudaMemcpyHostToDevice));
}

inline void initGpuMesh(GpuMesh *meshOut) {
	if (meshOut == nullptr) {
		return;
	}

	meshOut->positions_d = nullptr;
	meshOut->texcoords_d = nullptr;
	meshOut->normals_d = nullptr;
	meshOut->triangles_d = nullptr;
	meshOut->triNormals_d = nullptr;
	meshOut->vertexNormals_d = nullptr;
	meshOut->numPositions = 0;
	meshOut->numTexcoords = 0;
	meshOut->numNormals = 0;
	meshOut->numTriangles = 0;
	meshOut->ownsMemory = false;
}

inline void freeGpuMesh(GpuMesh *mesh) {
	if (mesh == nullptr) {
		return;
	}

	if (mesh->ownsMemory) {
		if (mesh->positions_d != nullptr) { CUDA_CHECK(cudaFree(mesh->positions_d)); }
		if (mesh->texcoords_d != nullptr) { CUDA_CHECK(cudaFree(mesh->texcoords_d)); }
		if (mesh->normals_d != nullptr) { CUDA_CHECK(cudaFree(mesh->normals_d)); }
		if (mesh->triangles_d != nullptr) { CUDA_CHECK(cudaFree(mesh->triangles_d)); }
		if (mesh->triNormals_d != nullptr) { CUDA_CHECK(cudaFree(mesh->triNormals_d)); }
		if (mesh->vertexNormals_d != nullptr) { CUDA_CHECK(cudaFree(mesh->vertexNormals_d)); }
	}

	initGpuMesh(mesh);
}

inline void uploadGpuMesh(GpuMesh *meshOut, const Mesh *mesh) {
	if (meshOut == nullptr || mesh == nullptr) {
		return;
	}

	freeGpuMesh(meshOut);
	uploadVector(&meshOut->positions_d, mesh->positions);
	uploadVector(&meshOut->texcoords_d, mesh->texcoords);
	uploadVector(&meshOut->normals_d, mesh->normals);
	uploadVector(&meshOut->triangles_d, mesh->triangles);
	uploadVector(&meshOut->triNormals_d, mesh->triNormals);
	uploadVector(&meshOut->vertexNormals_d, mesh->vertexNormals);

	meshOut->numPositions = static_cast<i32>(mesh->positions.size());
	meshOut->numTexcoords = static_cast<i32>(mesh->texcoords.size());
	meshOut->numNormals = static_cast<i32>(mesh->normals.size());
	meshOut->numTriangles = static_cast<i32>(mesh->triangles.size());
	meshOut->ownsMemory = true;
}

} // namespace geom
