#pragma once

/**
 * @file geometry_render.cuh
 * @brief Minimal CUDA mesh raycast preview renderer and PPM writer.
 */

#include "geometry_bvh.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

namespace geom {

struct MeshPreviewCamera {
	vec3 pos;
	vec3 target;
	vec3 up;
	f32 fovYRadians;
};

enum class MeshPreviewShadeMode : i32 {
	LitColor,
	Gooch,
	Normal,
	FaceNormal,
	Position,
};

struct MeshPreviewObject {
	const vec3 *positions;
	const MeshTri *triangles;
	i32 numPositions;
	i32 numTriangles;
	vec3 color;
	bool cullBackfaces;
};

struct MeshPreviewParams {
	i32 width;
	i32 height;
	MeshPreviewCamera camera;
	vec3 background;
	vec3 lightDir;
	f32 tMax;
	MeshPreviewShadeMode shadeMode;
	vec3 falseColorMin;
	vec3 falseColorMax;
};

struct MeshPreviewDeviceObject {
	MeshView mesh;
	BvhView bvh;
	vec3 color;
	bool cullBackfaces;
	bool isValid;
};

struct MeshPreviewSceneObject {
	Mesh mesh;
	Bvh bvh;
	GpuMesh gpuMesh;
	GpuBvh gpuBvh;
	MeshPreviewDeviceObject deviceObject;
	vec3 color;
	bool cullBackfaces;
	bool isValid;
};

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
u32 packRgb8(vec3 color) {
	const u32 r = u32(sprClampf(color.x, 0.0f, 1.0f)*255.0f + 0.5f);
	const u32 g = u32(sprClampf(color.y, 0.0f, 1.0f)*255.0f + 0.5f);
	const u32 b = u32(sprClampf(color.z, 0.0f, 1.0f)*255.0f + 0.5f);
	return r | (g << 8) | (b << 16);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 previewTanf(f32 x) {
	return tanf(x);
}

inline MeshPreviewObject makeMeshPreviewObject(
	const vec3 *positions,
	i32 numPositions,
	const MeshTri *triangles,
	i32 numTriangles,
	vec3 color,
	bool cullBackfaces = true) {
	MeshPreviewObject object = {};
	object.positions = positions;
	object.numPositions = numPositions;
	object.triangles = triangles;
	object.numTriangles = numTriangles;
	object.color = color;
	object.cullBackfaces = cullBackfaces;
	return object;
}

inline MeshPreviewObject makeMeshPreviewObject(
	const Mesh *mesh,
	vec3 color,
	bool cullBackfaces = true) {
	if (mesh == nullptr) {
		return {};
	}

	return makeMeshPreviewObject(
		mesh->positions.empty() ? nullptr : mesh->positions.data(),
		i32(mesh->positions.size()),
		mesh->triangles.empty() ? nullptr : mesh->triangles.data(),
		i32(mesh->triangles.size()),
		color,
		cullBackfaces);
}

inline MeshPreviewObject makeMeshPreviewObject(
	const std::vector<vec3> &positions,
	const std::vector<MeshTri> &triangles,
	vec3 color,
	bool cullBackfaces = true) {
	return makeMeshPreviewObject(
		positions.empty() ? nullptr : positions.data(),
		i32(positions.size()),
		triangles.empty() ? nullptr : triangles.data(),
		i32(triangles.size()),
		color,
		cullBackfaces);
}

inline Aabb computePreviewBounds(const MeshPreviewObject *objects, i32 numObjects) {
	Aabb bounds = emptyAabb();
	if (objects == nullptr) {
		return bounds;
	}

	for (i32 objectIdx = 0; objectIdx < numObjects; ++objectIdx) {
		const MeshPreviewObject object = objects[objectIdx];
		if (object.positions == nullptr) {
			continue;
		}

		for (i32 i = 0; i < object.numPositions; ++i) {
			growAabbPoint(&bounds, object.positions[i]);
		}
	}

	return bounds;
}

inline MeshPreviewCamera makeOrbitPreviewCamera(
	Aabb bounds,
	vec3 fromDirection,
	vec3 up = vec3(0.0f, 0.0f, 1.0f),
	f32 fovYRadians = 0.82f,
	f32 distanceScale = 2.25f) {
	MeshPreviewCamera camera = {};
	const vec3 center = isValidAabb(bounds) ? aabbCentroid(bounds) : vec3(0.0f);
	const vec3 extent = isValidAabb(bounds) ? aabbExtent(bounds) : vec3(1.0f);
	const f32 radius = sprMax<f32>(0.5f*length(extent), 1.0e-3f);
	distanceScale = sprMax<f32>(distanceScale, 1.25f);

	if (sqNorm(fromDirection) < 1.0e-8f) {
		fromDirection = vec3(2.5f, -3.0f, 1.6f);
	}
	fromDirection = normal(fromDirection);

	camera.target = center;
	camera.pos = center + fromDirection*(distanceScale*radius);
	camera.up = up;
	camera.fovYRadians = fovYRadians;
	return camera;
}

inline MeshPreviewParams defaultMeshPreviewParams(i32 width, i32 height) {
	MeshPreviewParams params = {};
	params.width = width;
	params.height = height;
	params.camera.pos = vec3(3.0f, -4.0f, 2.0f);
	params.camera.target = vec3(0.0f);
	params.camera.up = vec3(0.0f, 0.0f, 1.0f);
	params.camera.fovYRadians = 0.82f;
	params.background = vec3(0.025f, 0.028f, 0.032f);
	params.lightDir = normal(vec3(-0.35f, -0.55f, 0.9f));
	params.tMax = F32_MAX;
	params.shadeMode = MeshPreviewShadeMode::Gooch;
	params.falseColorMin = vec3(-1.0f);
	params.falseColorMax = vec3(1.0f);
	return params;
}

inline bool buildPreviewScene(
	std::vector<MeshPreviewSceneObject> *sceneOut,
	const MeshPreviewObject *objects,
	i32 numObjects) {
	if (sceneOut == nullptr || objects == nullptr || numObjects <= 0) {
		return false;
	}

	sceneOut->clear();
	sceneOut->resize(size_t(numObjects));
	bool hasValidObject = false;

	for (i32 objectIdx = 0; objectIdx < numObjects; ++objectIdx) {
		const MeshPreviewObject object = objects[objectIdx];
		MeshPreviewSceneObject &sceneObject = (*sceneOut)[size_t(objectIdx)];
		initGpuMesh(&sceneObject.gpuMesh);
		initGpuBvh(&sceneObject.gpuBvh);
		sceneObject.deviceObject = {};
		sceneObject.color = object.color;
		sceneObject.cullBackfaces = object.cullBackfaces;
		sceneObject.isValid = false;

		if (
			object.positions == nullptr ||
			object.triangles == nullptr ||
			object.numPositions <= 0 ||
			object.numTriangles <= 0) {
			continue;
		}

		sceneObject.mesh.positions.assign(object.positions, object.positions + object.numPositions);
		sceneObject.mesh.triangles.assign(object.triangles, object.triangles + object.numTriangles);
		ensureMeshNormals(&sceneObject.mesh);
		buildBvh(&sceneObject.bvh, &sceneObject.mesh);
		sceneObject.isValid = sceneObject.bvh.root >= 0;
		hasValidObject = hasValidObject || sceneObject.isValid;
	}

	return hasValidObject;
}

inline void freePreviewScene(std::vector<MeshPreviewSceneObject> *scene) {
	if (scene == nullptr) {
		return;
	}

	for (MeshPreviewSceneObject &object : *scene) {
		freeGpuMesh(&object.gpuMesh);
		freeGpuBvh(&object.gpuBvh);
	}
	scene->clear();
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void computePreviewBasis(
	vec3 *rightOut,
	vec3 *upOut,
	vec3 *forwardOut,
	const MeshPreviewCamera &camera) {
	vec3 forward = camera.target - camera.pos;
	if (sqNorm(forward) < 1.0e-8f) {
		forward = vec3(0.0f, 1.0f, 0.0f);
	}
	forward = normal(forward);

	vec3 right = cross(forward, camera.up);
	if (sqNorm(right) < 1.0e-8f) {
		right = cross(forward, vec3(0.0f, 0.0f, 1.0f));
	}
	if (sqNorm(right) < 1.0e-8f) {
		right = vec3(1.0f, 0.0f, 0.0f);
	}
	right = normal(right);
	const vec3 up = normal(cross(right, forward));

	*rightOut = right;
	*upOut = up;
	*forwardOut = forward;
}

inline MeshPreviewCamera makeFramedPreviewCamera(
	Aabb bounds,
	vec3 fromDirection,
	i32 width,
	i32 height,
	vec3 up = vec3(0.0f, 0.0f, 1.0f),
	f32 fovYRadians = 0.82f,
	f32 marginScale = 1.12f) {
	MeshPreviewCamera camera = makeOrbitPreviewCamera(
		bounds,
		fromDirection,
		up,
		fovYRadians,
		1.25f);
	if (!isValidAabb(bounds)) {
		return camera;
	}

	width = sprMax<i32>(width, 1);
	height = sprMax<i32>(height, 1);
	marginScale = sprMax<f32>(marginScale, 1.0f);

	vec3 right;
	vec3 cameraUp;
	vec3 forward;
	computePreviewBasis(&right, &cameraUp, &forward, camera);

	const vec3 center = aabbCentroid(bounds);
	const vec3 lo = bounds.lower;
	const vec3 hi = bounds.upper;
	const f32 aspect = f32(width)/f32(height);
	const f32 tanHalfY = sprMax<f32>(previewTanf(0.5f*fovYRadians), 1.0e-4f);
	const f32 tanHalfX = sprMax<f32>(tanHalfY*aspect, 1.0e-4f);

	f32 requiredDistance = 1.0e-3f;
	for (i32 corner = 0; corner < 8; ++corner) {
		const vec3 p(
			(corner & 1) ? hi.x : lo.x,
			(corner & 2) ? hi.y : lo.y,
			(corner & 4) ? hi.z : lo.z);
		const vec3 rel = p - center;
		const f32 x = sprAbs<f32>(dot(rel, right));
		const f32 y = sprAbs<f32>(dot(rel, cameraUp));
		const f32 z = dot(rel, forward);
		requiredDistance = sprMax<f32>(requiredDistance, x/tanHalfX - z);
		requiredDistance = sprMax<f32>(requiredDistance, y/tanHalfY - z);
		requiredDistance = sprMax<f32>(requiredDistance, 1.0e-3f - z);
	}

	camera.target = center;
	camera.pos = center - forward*(requiredDistance*marginScale);
	camera.up = cameraUp;
	camera.fovYRadians = fovYRadians;
	return camera;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
Ray makePreviewRay(
	const MeshPreviewParams &params,
	i32 px,
	i32 py,
	vec3 right,
	vec3 up,
	vec3 forward) {
	const f32 sx = 2.0f*(f32(px) + 0.5f)/f32(params.width) - 1.0f;
	const f32 sy = 1.0f - 2.0f*(f32(py) + 0.5f)/f32(params.height);
	const f32 aspect = f32(params.width)/f32(params.height);
	const f32 tanHalf = previewTanf(0.5f*params.camera.fovYRadians);
	const vec3 dir = normal(forward + right*(sx*tanHalf*aspect) + up*(sy*tanHalf));
	return makeRay(params.camera.pos, dir, 1.0e-4f, params.tMax);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 shadePreviewHit(vec3 baseColor, vec3 n, vec3 rayDir, vec3 lightDir) {
	if (dot(n, rayDir) > 0.0f) {
		n = -n;
	}

	if (sqNorm(lightDir) < 1.0e-8f) {
		lightDir = vec3(-0.35f, -0.55f, 0.9f);
	}
	lightDir = normal(lightDir);

	const f32 diffuse = sprFmaxf(0.0f, dot(n, lightDir));
	const f32 shade = 0.22f + 0.76f*diffuse;
	return baseColor*shade;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 shadePreviewGooch(vec3 baseColor, vec3 n, vec3 rayDir, vec3 lightDir) {
	if (dot(n, rayDir) > 0.0f) {
		n = -n;
	}

	if (sqNorm(lightDir) < 1.0e-8f) {
		lightDir = vec3(-0.35f, -0.55f, 0.9f);
	}
	lightDir = normal(lightDir);

	const f32 t = 0.5f*(dot(n, lightDir) + 1.0f);
	const vec3 cool = vec3(0.10f, 0.24f, 0.48f) + 0.34f*baseColor;
	const vec3 warm = vec3(0.92f, 0.74f, 0.32f) + 0.28f*baseColor;
	vec3 color = cool*(1.0f - t) + warm*t;

	const f32 facing = sprAbs<f32>(dot(n, rayDir));
	const f32 rim = sprClampf((0.34f - facing)/0.34f, 0.0f, 1.0f);
	color = color*(1.0f - 0.28f*rim);
	return color;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 falseColorPreviewPosition(vec3 p, vec3 lo, vec3 hi) {
	const vec3 extent = hi - lo;
	vec3 color(0.5f);
	if (sprAbs<f32>(extent.x) > 1.0e-8f) {
		color.x = sprClampf((p.x - lo.x)/extent.x, 0.0f, 1.0f);
	}
	if (sprAbs<f32>(extent.y) > 1.0e-8f) {
		color.y = sprClampf((p.y - lo.y)/extent.y, 0.0f, 1.0f);
	}
	if (sprAbs<f32>(extent.z) > 1.0e-8f) {
		color.z = sprClampf((p.z - lo.z)/extent.z, 0.0f, 1.0f);
	}
	return color;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 previewShadingNormal(MeshView mesh, RayHit hit) {
	const vec3 faceNormal = hit.normal;
	if (
		mesh.vertexNormals == nullptr ||
		mesh.triangles == nullptr ||
		hit.triIdx < 0 ||
		hit.triIdx >= mesh.numTriangles) {
		return faceNormal;
	}

	const MeshTri tri = mesh.triangles[hit.triIdx];
	if (
		tri.v0 < 0 || tri.v0 >= mesh.numPositions ||
		tri.v1 < 0 || tri.v1 >= mesh.numPositions ||
		tri.v2 < 0 || tri.v2 >= mesh.numPositions) {
		return faceNormal;
	}

	vec3 n =
		hit.u*mesh.vertexNormals[tri.v0] +
		hit.v*mesh.vertexNormals[tri.v1] +
		hit.w*mesh.vertexNormals[tri.v2];
	if (sqNorm(n) < 1.0e-10f) {
		return faceNormal;
	}

	n = normal(n);
	return dot(n, faceNormal) < 0.0f ? -n : n;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 shadePreviewSample(vec3 baseColor, RayHit hit, vec3 rayDir, const MeshPreviewParams &params) {
	if (params.shadeMode == MeshPreviewShadeMode::Normal) {
		const vec3 n = hit.normal;
		return vec3(
			0.5f*n.x + 0.5f,
			0.5f*n.y + 0.5f,
			0.5f*n.z + 0.5f);
	}

	if (params.shadeMode == MeshPreviewShadeMode::FaceNormal) {
		const vec3 n = hit.normal;
		return vec3(
			0.5f*n.x + 0.5f,
			0.5f*n.y + 0.5f,
			0.5f*n.z + 0.5f);
	}

	if (params.shadeMode == MeshPreviewShadeMode::Position) {
		return falseColorPreviewPosition(hit.position, params.falseColorMin, params.falseColorMax);
	}

	if (params.shadeMode == MeshPreviewShadeMode::Gooch) {
		return shadePreviewGooch(baseColor, hit.normal, rayDir, params.lightDir);
	}

	return shadePreviewHit(baseColor, hit.normal, rayDir, params.lightDir);
}

static __global__ void renderMeshPreviewPixels(
	u32 *pixelsOut_d,
	const MeshPreviewDeviceObject *objects,
	i32 numObjects,
	MeshPreviewParams params,
	vec3 right,
	vec3 up,
	vec3 forward) {
	const i32 pixelIdx = i32(blockIdx.x*blockDim.x + threadIdx.x);
	const i32 numPixels = params.width*params.height;
	if (pixelIdx >= numPixels) {
		return;
	}

	const i32 px = pixelIdx%params.width;
	const i32 py = pixelIdx/params.width;
	const Ray ray = makePreviewRay(params, px, py, right, up, forward);
	RayHit bestHit = makeInvalidRayHit();
	vec3 bestColor(0.0f);
	bool hasHit = false;

	for (i32 objectIdx = 0; objectIdx < numObjects; ++objectIdx) {
		const MeshPreviewDeviceObject object = objects[objectIdx];
		if (!object.isValid) {
			continue;
		}

		RayHit hit;
		if (!raycastBvh(&hit, object.mesh, object.bvh, ray)) {
			continue;
		}

		const vec3 faceNormal = hit.normal;
		if (object.cullBackfaces && dot(faceNormal, ray.dir) >= 0.0f) {
			continue;
		}
		if (params.shadeMode != MeshPreviewShadeMode::FaceNormal) {
			hit.normal = previewShadingNormal(object.mesh, hit);
		}

		if (!hasHit || hit.t < bestHit.t) {
			bestHit = hit;
			bestColor = object.color;
			hasHit = true;
		}
	}

	vec3 color = params.background;
	if (hasHit) {
		color = shadePreviewSample(bestColor, bestHit, ray.dir, params);
	}
	pixelsOut_d[pixelIdx] = packRgb8(color);
}

inline bool writePreviewPpm(const char *path, const std::vector<u32> &pixels, i32 width, i32 height) {
	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	std::fprintf(file, "P6\n%d %d\n255\n", width, height);
	for (u32 pixel : pixels) {
		const u8 rgb[3] = {
			u8(pixel & 0xffu),
			u8((pixel >> 8) & 0xffu),
			u8((pixel >> 16) & 0xffu),
		};
		std::fwrite(rgb, 1, 3, file);
	}

	std::fclose(file);
	return true;
}

inline bool renderMeshPreviewPpm(
	const char *path,
	const MeshPreviewObject *objects,
	i32 numObjects,
	MeshPreviewParams params) {
	if (path == nullptr || objects == nullptr || numObjects <= 0 || params.width <= 0 || params.height <= 0) {
		return false;
	}

	std::vector<MeshPreviewSceneObject> scene;
	if (!buildPreviewScene(&scene, objects, numObjects)) {
		return false;
	}

	vec3 right;
	vec3 up;
	vec3 forward;
	computePreviewBasis(&right, &up, &forward, params.camera);

	std::vector<MeshPreviewDeviceObject> deviceObjects(static_cast<size_t>(numObjects));
	for (i32 objectIdx = 0; objectIdx < numObjects; ++objectIdx) {
		MeshPreviewSceneObject &object = scene[size_t(objectIdx)];
		if (!object.isValid) {
			deviceObjects[size_t(objectIdx)] = {};
			continue;
		}

		uploadGpuMesh(&object.gpuMesh, &object.mesh);
		uploadGpuBvh(&object.gpuBvh, &object.bvh);

		object.deviceObject.mesh = viewGpuMesh(&object.gpuMesh);
		object.deviceObject.bvh = viewGpuBvh(&object.gpuBvh);
		object.deviceObject.color = object.color;
		object.deviceObject.cullBackfaces = object.cullBackfaces;
		object.deviceObject.isValid = true;
		deviceObjects[size_t(objectIdx)] = object.deviceObject;
	}

	MeshPreviewDeviceObject *objects_d = nullptr;
	u32 *pixels_d = nullptr;
	const size_t numPixels = static_cast<size_t>(params.width)*static_cast<size_t>(params.height);
	std::vector<u32> pixels(numPixels, packRgb8(params.background));

	uploadVector(&objects_d, deviceObjects);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pixels_d), numPixels*sizeof(u32)));

	const i32 threadsPerBlock = 256;
	const i32 numBlocks = dbg::divUp(i32(numPixels), threadsPerBlock);
	renderMeshPreviewPixels<<<numBlocks, threadsPerBlock>>>(
		pixels_d,
		objects_d,
		numObjects,
		params,
		right,
		up,
		forward);
	CUDA_LAUNCH_CHECK();
	CUDA_CHECK(cudaMemcpy(pixels.data(), pixels_d, numPixels*sizeof(u32), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(pixels_d));
	CUDA_CHECK(cudaFree(objects_d));
	freePreviewScene(&scene);

	return writePreviewPpm(path, pixels, params.width, params.height);
}

} // namespace geom
