#pragma once

/**
 * @file geometry_obj.cuh
 * @brief Minimal OBJ loader for v/vt/vn/f records.
 */

#include "geometry_mesh.cuh"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace geom {

struct ObjFaceVertex {
	i32 v;
	i32 vt;
	i32 vn;
};

static inline i32 convertObjIndex(i32 objIdx, i32 count) {
	if (objIdx > 0) {
		return objIdx - 1;
	}
	if (objIdx < 0) {
		return count + objIdx;
	}
	return GEOM_INVALID_IDX;
}

static inline bool parseObjFaceVertex(
	ObjFaceVertex *out,
	const char *token,
	i32 numPositions,
	i32 numTexcoords,
	i32 numNormals) {
	if (out == nullptr || token == nullptr || token[0] == '\0') {
		return false;
	}

	out->v = GEOM_INVALID_IDX;
	out->vt = GEOM_INVALID_IDX;
	out->vn = GEOM_INVALID_IDX;

	char *end = nullptr;
	long rawV = std::strtol(token, &end, 10);
	if (end == token) {
		return false;
	}
	out->v = convertObjIndex(static_cast<i32>(rawV), numPositions);
	if (out->v < 0 || out->v >= numPositions) {
		return false;
	}

	const char *cursor = end;
	if (*cursor != '/') {
		return true;
	}

	++cursor;
	if (*cursor != '/' && *cursor != '\0') {
		char *vtEnd = nullptr;
		long rawVt = std::strtol(cursor, &vtEnd, 10);
		if (vtEnd != cursor) {
			out->vt = convertObjIndex(static_cast<i32>(rawVt), numTexcoords);
			cursor = vtEnd;
		}
	}

	if (*cursor != '/') {
		return true;
	}

	++cursor;
	if (*cursor != '\0') {
		char *vnEnd = nullptr;
		long rawVn = std::strtol(cursor, &vnEnd, 10);
		if (vnEnd != cursor) {
			out->vn = convertObjIndex(static_cast<i32>(rawVn), numNormals);
		}
	}

	return true;
}

static inline void appendObjTriangle(Mesh *mesh, ObjFaceVertex a, ObjFaceVertex b, ObjFaceVertex c) {
	MeshTri tri = {};
	tri.v0 = a.v;
	tri.v1 = b.v;
	tri.v2 = c.v;
	tri.vt0 = a.vt;
	tri.vt1 = b.vt;
	tri.vt2 = c.vt;
	tri.vn0 = a.vn;
	tri.vn1 = b.vn;
	tri.vn2 = c.vn;
	mesh->triangles.push_back(tri);
}

static inline void normalizeMeshToUnitBox(Mesh *mesh, bool shouldCenter, bool shouldNormalizeToUnitBox) {
	if (mesh == nullptr || mesh->positions.empty()) {
		return;
	}
	if (!shouldCenter && !shouldNormalizeToUnitBox) {
		return;
	}

	const Aabb bounds = computeMeshBounds(mesh);
	if (!isValidAabb(bounds)) {
		return;
	}

	const vec3 center = aabbCentroid(bounds);
	const vec3 extent = aabbExtent(bounds);
	const f32 maxExtent = sprMax<f32>(extent.x, sprMax<f32>(extent.y, extent.z));
	const f32 scale = (shouldNormalizeToUnitBox && maxExtent > 0.0f) ? (2.0f/maxExtent) : 1.0f;

	for (vec3 &p : mesh->positions) {
		if (shouldCenter) {
			p -= center;
		}
		p *= scale;
	}
}

[[nodiscard]] inline bool tryLoadObj(Mesh *meshOut, const char *path, ObjLoadOptions options) {
	if (meshOut == nullptr || path == nullptr) {
		return false;
	}

	FILE *file = std::fopen(path, "rb");
	if (file == nullptr) {
		if (options.shouldPrintWarnings) {
			std::fprintf(stderr, "geom: failed to open OBJ '%s'\n", path);
		}
		return false;
	}

	Mesh mesh;
	char line[4096];
	i32 lineNumber = 0;
	bool ok = true;

	while (std::fgets(line, sizeof(line), file) != nullptr) {
		++lineNumber;
		char *cursor = line;
		while (*cursor != '\0' && std::isspace(static_cast<unsigned char>(*cursor))) {
			++cursor;
		}

		if (*cursor == '\0' || *cursor == '#') {
			continue;
		}

		if (cursor[0] == 'v' && std::isspace(static_cast<unsigned char>(cursor[1]))) {
			f32 x = 0.0f;
			f32 y = 0.0f;
			f32 z = 0.0f;
			if (std::sscanf(cursor + 1, "%f %f %f", &x, &y, &z) == 3) {
				mesh.positions.push_back(vec3(x, y, z));
			}
			continue;
		}

		if (cursor[0] == 'v' && cursor[1] == 't' && std::isspace(static_cast<unsigned char>(cursor[2]))) {
			f32 u = 0.0f;
			f32 v = 0.0f;
			if (std::sscanf(cursor + 2, "%f %f", &u, &v) >= 2) {
				mesh.texcoords.push_back(vec2(u, v));
			}
			continue;
		}

		if (cursor[0] == 'v' && cursor[1] == 'n' && std::isspace(static_cast<unsigned char>(cursor[2]))) {
			f32 x = 0.0f;
			f32 y = 0.0f;
			f32 z = 0.0f;
			if (std::sscanf(cursor + 2, "%f %f %f", &x, &y, &z) == 3) {
				vec3 n(x, y, z);
				normalize(n);
				mesh.normals.push_back(n);
			}
			continue;
		}

		if (cursor[0] == 'f' && std::isspace(static_cast<unsigned char>(cursor[1]))) {
			std::vector<ObjFaceVertex> face;
			char *token = std::strtok(cursor + 1, " \t\r\n");
			while (token != nullptr) {
				ObjFaceVertex fv;
				if (parseObjFaceVertex(
					&fv,
					token,
					static_cast<i32>(mesh.positions.size()),
					static_cast<i32>(mesh.texcoords.size()),
					static_cast<i32>(mesh.normals.size()))) {
					face.push_back(fv);
				}
				token = std::strtok(nullptr, " \t\r\n");
			}

			if (face.size() < 3) {
				if (options.shouldPrintWarnings) {
					std::fprintf(stderr, "geom: OBJ line %d has a face with fewer than 3 valid vertices\n", lineNumber);
				}
				ok = false;
				continue;
			}

			if (!options.shouldTriangulate && face.size() != 3) {
				if (options.shouldPrintWarnings) {
					std::fprintf(stderr, "geom: OBJ line %d is non-triangle and triangulation is disabled\n", lineNumber);
				}
				ok = false;
				continue;
			}

			for (size_t i = 1; i + 1 < face.size(); ++i) {
				appendObjTriangle(&mesh, face[0], face[i], face[i + 1]);
			}
			continue;
		}
	}

	std::fclose(file);

	normalizeMeshToUnitBox(&mesh, options.shouldCenter, options.shouldNormalizeToUnitBox);
	if (options.shouldComputeMissingNormals) {
		ensureMeshNormals(&mesh);
	}

	ok = validateMesh(&mesh, options.shouldPrintWarnings) && ok;
	if (!ok) {
		return false;
	}

	*meshOut = std::move(mesh);
	return true;
}

[[nodiscard]] inline bool tryLoadObj(Mesh *meshOut, const char *path) {
	return tryLoadObj(meshOut, path, defaultObjLoadOptions());
}

[[nodiscard]] inline bool tryWriteObj(
	const char *path,
	const vec3 *positions,
	i32 numPositions,
	const MeshTri *triangles,
	i32 numTriangles) {
	if (
		path == nullptr ||
		positions == nullptr ||
		triangles == nullptr ||
		numPositions <= 0 ||
		numTriangles <= 0) {
		return false;
	}

	Mesh mesh;
	mesh.positions.assign(positions, positions + numPositions);
	mesh.triangles.assign(triangles, triangles + numTriangles);
	ensureMeshNormals(&mesh);

	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	for (vec3 p : mesh.positions) {
		std::fprintf(file, "v %.9g %.9g %.9g\n", p.x, p.y, p.z);
	}

	const bool hasVertexNormals = mesh.vertexNormals.size() == mesh.positions.size();
	if (hasVertexNormals) {
		for (vec3 n : mesh.vertexNormals) {
			std::fprintf(file, "vn %.9g %.9g %.9g\n", n.x, n.y, n.z);
		}
	}

	bool ok = true;
	for (MeshTri tri : mesh.triangles) {
		const bool triValid =
			tri.v0 >= 0 && tri.v0 < numPositions &&
			tri.v1 >= 0 && tri.v1 < numPositions &&
			tri.v2 >= 0 && tri.v2 < numPositions;
		if (!triValid) {
			ok = false;
			continue;
		}

		if (hasVertexNormals) {
			std::fprintf(
				file,
				"f %d//%d %d//%d %d//%d\n",
				tri.v0 + 1,
				tri.v0 + 1,
				tri.v1 + 1,
				tri.v1 + 1,
				tri.v2 + 1,
				tri.v2 + 1);
		} else {
			std::fprintf(file, "f %d %d %d\n", tri.v0 + 1, tri.v1 + 1, tri.v2 + 1);
		}
	}

	std::fclose(file);
	return ok;
}

[[nodiscard]] inline bool tryWriteObj(
	const char *path,
	const std::vector<vec3> &positions,
	const std::vector<MeshTri> &triangles) {
	return tryWriteObj(
		path,
		positions.empty() ? nullptr : positions.data(),
		static_cast<i32>(positions.size()),
		triangles.empty() ? nullptr : triangles.data(),
		static_cast<i32>(triangles.size()));
}

[[nodiscard]] inline bool tryWriteObj(const char *path, const Mesh *mesh) {
	if (mesh == nullptr) {
		return false;
	}

	return tryWriteObj(path, mesh->positions, mesh->triangles);
}

[[nodiscard]] inline bool tryWriteObj(const char *path, const Mesh &mesh) {
	return tryWriteObj(path, &mesh);
}

} // namespace geom
