#include "mesh_to_adsdf.cuh"
#include "experiment_diagnostics.cuh"
#include "numeric.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


#define TENTACLE_OUTPUT_PREFIX "tentacle_armature_adsdf"
#define TENTACLE_REPORT_LABEL "ADSDF tentacle"
using ObstacleSample = adsdf::AdsdfPointSample;
using ObstacleSampler = adsdf::GpuAdsdfSampler;

struct ObstacleField {
	adsdf::AdsdfView field;
};

struct ExperimentParams {
	f32 desiredClearance = 0.055f;
	f32 softplusEps = 0.020f;

	f32 tempClear = 360.0f;
	f32 tempArap = 0.08f;
	f32 tempGuide = 0.020f;
	f32 tempControlBend = 0.04f;

	i32 iterations = 300;
	f32 controlStep = 0.22f;
	f32 rotationStep = 0.016f;
	expdiag::StopCriteria stop;
};

struct ControlPoint {
	vec3 position;
	vec3 guide;
	bool isPinned = false;
};

static constexpr i32 MAX_TENTACLE_INFLUENCES = 6;

struct VertexBinding {
	i32 section;
	i32 numInfluences;
	i32 controls[MAX_TENTACLE_INFLUENCES];
	f32 weights[MAX_TENTACLE_INFLUENCES];
	f32 u;
	f32 theta;
	f32 radius;
};

struct TentacleState {
	std::vector<ControlPoint> controls;
	std::vector<VertexBinding> bindings;
	std::vector<vec3> p0;
	std::vector<vec3> p;
	std::vector<quat> q;
	std::vector<f32> vertexAreas;
	std::vector<f32> controlMass;
	geom::MeshTopology topology;
	i32 sections = 0;
	i32 radialSegments = 0;
};

struct EnergyReport {
	f64 total = 0.0;
	f64 clearance = 0.0;
	f64 arap = 0.0;
	f64 guide = 0.0;
	f64 controlBend = 0.0;

	f32 minPhi = F32_MAX;
	f32 maxViolation = 0.0f;
	f32 avgViolation = 0.0f;
	f32 rmsGuide = 0.0f;
	f32 maxControlDisplacement = 0.0f;
	i32 numViolating = 0;
};

static expdiag::IterationSample makeDiagnosticSample(i32 iteration, const EnergyReport &report) {
	expdiag::IterationSample sample = {};
	sample.iteration = iteration;
	sample.energy = report.total;
	sample.maxViolation = report.maxViolation;
	sample.avgViolation = report.avgViolation;
	sample.minPhi = report.minPhi;
	sample.rmsGuide = report.rmsGuide;
	sample.maxControl = report.maxControlDisplacement;
	sample.numViolating = report.numViolating;
	return sample;
}

struct ControlLmStats {
	i32 iterations = 0;
	i32 accepted = 0;
	i32 rejected = 0;
	f64 beforeEnergy = 0.0;
	f64 afterEnergy = 0.0;
	f64 beforeControlObjective = 0.0;
	f64 afterControlObjective = 0.0;
};

struct OrientationReport {
	i32 numFaces = 0;
	i32 numOutward = 0;
	i32 numInward = 0;
	f32 minDot = F32_MAX;
	f32 avgDot = 0.0f;
};

static f32 stableSoftplus(f32 t, f32 eps) {
	const f32 x = t/eps;
	if (x > 20.0f) {
		return t;
	}
	if (x < -20.0f) {
		return eps*std::exp(x);
	}
	return eps*std::log1p(std::exp(x));
}

static f32 stableSigmoid(f32 t, f32 eps) {
	const f32 x = t/eps;
	if (x > 20.0f) {
		return 1.0f;
	}
	if (x < -20.0f) {
		return std::exp(x);
	}
	return 1.0f/(1.0f + std::exp(-x));
}

static void appendEllipsoidMesh(
	geom::Mesh *mesh,
	vec3 center,
	vec3 scale,
	i32 rings,
	i32 segments) {
	const geom::vid baseIdx = geom::vid(mesh->positions.size());

	auto dirFromAngles = [](f32 theta, f32 phi) {
		const f32 s = sprSinf(phi);
		return vec3(s*sprCosf(theta), s*sprSinf(theta), sprCosf(phi));
	};

	auto place = [center, scale](vec3 dir) {
		return center + vec3(scale.x*dir.x, scale.y*dir.y, scale.z*dir.z);
	};

	mesh->positions.push_back(place(vec3(0.0f, 0.0f, 1.0f)));

	for (i32 r = 1; r < rings; ++r) {
		const f32 phi = f32(SPR_PI)*f32(r)/f32(rings);
		for (i32 s = 0; s < segments; ++s) {
			const f32 theta = 2.0f*f32(SPR_PI)*f32(s)/f32(segments);
			mesh->positions.push_back(place(dirFromAngles(theta, phi)));
		}
	}

	const geom::vid bottomIdx = geom::vid(mesh->positions.size());
	mesh->positions.push_back(place(vec3(0.0f, 0.0f, -1.0f)));

	auto ringVertex = [baseIdx, segments](i32 r, i32 s) {
		const i32 wrapped = (s % segments + segments) % segments;
		return geom::vid(baseIdx + 1 + (r - 1)*segments + wrapped);
	};

	for (i32 s = 0; s < segments; ++s) {
		geom::appendTriangle(mesh, baseIdx, ringVertex(1, s), ringVertex(1, s + 1));
	}

	for (i32 r = 1; r + 1 < rings; ++r) {
		for (i32 s = 0; s < segments; ++s) {
			const geom::vid v00 = ringVertex(r, s);
			const geom::vid v01 = ringVertex(r, s + 1);
			const geom::vid v10 = ringVertex(r + 1, s);
			const geom::vid v11 = ringVertex(r + 1, s + 1);
			geom::appendTriangle(mesh, v00, v10, v11);
			geom::appendTriangle(mesh, v00, v11, v01);
		}
	}

	for (i32 s = 0; s < segments; ++s) {
		geom::appendTriangle(mesh, ringVertex(rings - 1, s), bottomIdx, ringVertex(rings - 1, s + 1));
	}
}

static void makeObstacleMesh(geom::Mesh *meshOut) {
	geom::clearMesh(meshOut);

	appendEllipsoidMesh(meshOut, vec3(-0.82f, 0.02f, 0.02f), vec3(0.34f, 0.42f, 0.55f), 18, 32);
	appendEllipsoidMesh(meshOut, vec3(0.12f, -0.08f, 0.08f), vec3(0.40f, 0.34f, 0.48f), 18, 32);
	appendEllipsoidMesh(meshOut, vec3(0.90f, 0.10f, -0.04f), vec3(0.32f, 0.44f, 0.46f), 18, 32);
	appendEllipsoidMesh(meshOut, vec3(0.40f, 0.38f, -0.18f), vec3(0.26f, 0.20f, 0.34f), 14, 24);

	geom::ensureMeshNormals(meshOut);
}

static bool writeObj(const char *path, const std::vector<vec3> &positions, const std::vector<geom::MeshTri> &triangles) {
	return geom::tryWriteObj(path, positions, triangles);
}

static bool writeObj(const char *path, const geom::Mesh &mesh) {
	return geom::tryWriteObj(path, mesh);
}

static bool writeControlObj(const char *path, const std::vector<ControlPoint> &controls, bool useGuide) {
	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	for (ControlPoint control : controls) {
		const vec3 p = useGuide ? control.guide : control.position;
		std::fprintf(file, "v %.9g %.9g %.9g\n", p.x, p.y, p.z);
	}

	if (!controls.empty()) {
		std::fprintf(file, "l");
		for (size_t i = 0; i < controls.size(); ++i) {
			std::fprintf(file, " %zu", i + 1);
		}
		std::fprintf(file, "\n");
	}

	std::fclose(file);
	return true;
}

static vec3 controlPointPosition(const ControlPoint &control, bool useGuide) {
	return useGuide ? control.guide : control.position;
}

static vec3 controlTangent(const std::vector<ControlPoint> &controls, i32 idx, bool useGuide) {
	const i32 n = i32(controls.size());
	if (n <= 1) {
		return vec3(1.0f, 0.0f, 0.0f);
	}

	const i32 prevIdx = sprMax<i32>(idx - 1, 0);
	const i32 nextIdx = sprMin<i32>(idx + 1, n - 1);
	vec3 tangent =
		controlPointPosition(controls[size_t(nextIdx)], useGuide) -
		controlPointPosition(controls[size_t(prevIdx)], useGuide);
	if (sqNorm(tangent) < 1.0e-10f) {
		tangent = vec3(1.0f, 0.0f, 0.0f);
	}
	return normal(tangent);
}

static vec3 safeFrameRight(vec3 tangent) {
	vec3 seed(0.0f, 0.0f, 1.0f);
	if (std::fabs(dot(seed, tangent)) > 0.85f) {
		seed = vec3(0.0f, 1.0f, 0.0f);
	}

	vec3 right = cross(seed, tangent);
	if (sqNorm(right) < 1.0e-10f) {
		right = vec3(0.0f, 1.0f, 0.0f);
	}
	return normal(right);
}

static geom::Mesh makeControlCurveMesh(
	const std::vector<ControlPoint> &controls,
	bool useGuide,
	f32 radius,
	i32 radialSegments) {
	geom::Mesh mesh;
	if (controls.size() < 2 || radius <= 0.0f || radialSegments < 3) {
		return mesh;
	}

	const i32 numControls = i32(controls.size());
	mesh.positions.reserve(size_t(numControls)*size_t(radialSegments) + 2);

	vec3 right(0.0f, 1.0f, 0.0f);
	for (i32 c = 0; c < numControls; ++c) {
		const vec3 p = controlPointPosition(controls[size_t(c)], useGuide);
		const vec3 tangent = controlTangent(controls, c, useGuide);
		if (c == 0) {
			right = safeFrameRight(tangent);
		} else {
			right = right - dot(right, tangent)*tangent;
			if (sqNorm(right) < 1.0e-10f) {
				right = safeFrameRight(tangent);
			} else {
				right = normal(right);
			}
		}
		const vec3 up = normal(cross(tangent, right));

		for (i32 r = 0; r < radialSegments; ++r) {
			const f32 theta = 2.0f*f32(SPR_PI)*f32(r)/f32(radialSegments);
			mesh.positions.push_back(p + radius*(sprCosf(theta)*right + sprSinf(theta)*up));
		}
	}

	for (i32 c = 0; c + 1 < numControls; ++c) {
		for (i32 r = 0; r < radialSegments; ++r) {
			const geom::vid v00 = c*radialSegments + r;
			const geom::vid v01 = c*radialSegments + ((r + 1)%radialSegments);
			const geom::vid v10 = (c + 1)*radialSegments + r;
			const geom::vid v11 = (c + 1)*radialSegments + ((r + 1)%radialSegments);
			geom::appendTriangle(&mesh, v00, v11, v10);
			geom::appendTriangle(&mesh, v00, v01, v11);
		}
	}

	const geom::vid startCenter = geom::vid(mesh.positions.size());
	mesh.positions.push_back(controlPointPosition(controls.front(), useGuide));
	for (i32 r = 0; r < radialSegments; ++r) {
		const geom::vid v0 = r;
		const geom::vid v1 = (r + 1)%radialSegments;
		geom::appendTriangle(&mesh, startCenter, v1, v0);
	}

	const geom::vid endCenter = geom::vid(mesh.positions.size());
	mesh.positions.push_back(controlPointPosition(controls.back(), useGuide));
	const geom::vid endBase = (numControls - 1)*radialSegments;
	for (i32 r = 0; r < radialSegments; ++r) {
		const geom::vid v0 = endBase + r;
		const geom::vid v1 = endBase + ((r + 1)%radialSegments);
		geom::appendTriangle(&mesh, endCenter, v0, v1);
	}

	geom::ensureMeshNormals(&mesh);
	return mesh;
}

static vec3 safeCameraRight(vec3 dir) {
	const vec3 worldUp(0.0f, 0.0f, 1.0f);
	vec3 right = cross(dir, worldUp);
	if (sqNorm(right) < 1.0e-8f) {
		right = cross(dir, vec3(0.0f, 1.0f, 0.0f));
	}
	return normal(right);
}

static adsdf::AdsdfCamera makeAdsdfCamera(
	vec3 center,
	f32 radius,
	i32 width,
	i32 height,
	vec3 fromDirection) {
	adsdf::AdsdfCamera camera = {};
	if (sqNorm(fromDirection) < 1.0e-8f) {
		fromDirection = vec3(2.5f, -2.7f, 1.2f);
	}

	camera.pos = center + normal(fromDirection)*(2.0f*radius);
	camera.dir = normal(center - camera.pos);
	camera.right = safeCameraRight(camera.dir);
	camera.up = normal(cross(camera.right, camera.dir));
	camera.tanHalfFovY = 0.42f;
	camera.aspect = f32(width)/f32(height);
	return camera;
}

static adsdf::AdsdfRayMarchParams makeRayMarchParams(i32 width, i32 height, f32 radius, adsdf::AdsdfCamera camera) {
	adsdf::AdsdfRayMarchParams params = {};
	params.imageWidth = width;
	params.imageHeight = height;
	params.maxSteps = 512;
	params.hitEps = sprMax<f32>(1.0e-4f*radius, 1.0e-4f);
	params.normalEps = sprMax<f32>(5.0e-4f*radius, 1.0e-4f);
	params.tMax = 8.0f*radius;
	params.stepScale = 0.75f;
	params.minStep = sprMax<f32>(1.0e-4f*radius, 1.0e-4f);
	params.maxStep = sprMax<f32>(0.05f*radius, 1.0e-3f);
	params.lightDir = normal(-camera.dir + camera.up*0.45f + camera.right*0.2f);
	return params;
}

static bool renderAdsdfPreviewPpm(
	const char *path,
	adsdf::AdsdfView field,
	vec3 center,
	f32 radius,
	vec3 fromDirection,
	i32 width,
	i32 height) {
	u32 *pixels_d = nullptr;
	const size_t pixelsSize_bytes = size_t(width)*size_t(height)*sizeof(u32);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pixels_d), pixelsSize_bytes));

	std::vector<u32> pixels_h(size_t(width)*size_t(height));
	const adsdf::AdsdfCamera camera = makeAdsdfCamera(center, radius, width, height, fromDirection);
	const adsdf::AdsdfRayMarchParams rayParams = makeRayMarchParams(width, height, radius, camera);

	adsdf::launchRender(pixels_d, field, camera, rayParams);
	CUDA_SYNC_CHECK();
	CUDA_CHECK(cudaMemcpy(pixels_h.data(), pixels_d, pixelsSize_bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(pixels_d));

	return adsdf::writePpm(path, pixels_h.data(), width, height);
}

static vec3 weightedControlPosition(const std::vector<ControlPoint> &controls, const VertexBinding &binding);

static vec3 hostCross(vec3 a, vec3 b) {
	return vec3(
		a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x);
}

static f32 hostDot(vec3 a, vec3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

static bool hostNormalize(vec3 *v) {
	const f64 len2 =
		double(v->x)*double(v->x) +
		double(v->y)*double(v->y) +
		double(v->z)*double(v->z);
	if (len2 <= 1.0e-20 || !std::isfinite(len2)) {
		return false;
	}

	const f32 invLen = f32(1.0/std::sqrt(len2));
	*v *= invLen;
	return true;
}

static OrientationReport evaluateTubeOrientation(
	const TentacleState &state,
	const std::vector<geom::MeshTri> &triangles) {
	OrientationReport report = {};
	if (state.p.empty() || state.bindings.size() != state.p.size()) {
		report.minDot = 0.0f;
		return report;
	}

	f64 dotSum = 0.0;
	for (geom::MeshTri tri : triangles) {
		if (
			tri.v0 < 0 || tri.v0 >= i32(state.p.size()) ||
			tri.v1 < 0 || tri.v1 >= i32(state.p.size()) ||
			tri.v2 < 0 || tri.v2 >= i32(state.p.size())) {
			continue;
		}

		const vec3 a = state.p[size_t(tri.v0)];
		const vec3 b = state.p[size_t(tri.v1)];
		const vec3 c = state.p[size_t(tri.v2)];
		vec3 n = hostCross(b - a, c - a);
		if (!hostNormalize(&n)) {
			continue;
		}
		vec3 radial =
			(state.p[size_t(tri.v0)] - weightedControlPosition(state.controls, state.bindings[size_t(tri.v0)])) +
			(state.p[size_t(tri.v1)] - weightedControlPosition(state.controls, state.bindings[size_t(tri.v1)])) +
			(state.p[size_t(tri.v2)] - weightedControlPosition(state.controls, state.bindings[size_t(tri.v2)]));
		if (!hostNormalize(&radial)) {
			continue;
		}

		const f32 d = hostDot(n, radial);
		report.minDot = std::min(report.minDot, d);
		dotSum += double(d);
		++report.numFaces;
		if (d >= 0.0f) {
			++report.numOutward;
		} else {
			++report.numInward;
		}
	}

	if (report.numFaces > 0) {
		report.avgDot = f32(dotSum/double(report.numFaces));
	} else {
		report.minDot = 0.0f;
	}
	return report;
}

static vec3 guideControlPosition(i32 idx, i32 numControls) {
	const f32 u = f32(idx)/f32(numControls - 1);
	const f32 x = -1.65f + 3.30f*u;
	const f32 y = 0.13f*sprSinf(2.0f*f32(SPR_PI)*u + 0.35f);
	const f32 z = 0.10f*sprCosf(1.5f*f32(SPR_PI)*u - 0.20f);
	return vec3(x, y, z);
}

static void addBindingWeight(VertexBinding *binding, i32 controlIdx, f32 weight, i32 numControls) {
	if (binding == nullptr || std::fabs(weight) <= 1.0e-7f) {
		return;
	}

	controlIdx = std::max(0, std::min(controlIdx, numControls - 1));
	for (i32 i = 0; i < binding->numInfluences; ++i) {
		if (binding->controls[i] == controlIdx) {
			binding->weights[i] += weight;
			return;
		}
	}

	if (binding->numInfluences >= MAX_TENTACLE_INFLUENCES) {
		return;
	}

	const i32 idx = binding->numInfluences++;
	binding->controls[idx] = controlIdx;
	binding->weights[idx] = weight;
}

static VertexBinding makeVertexBinding(i32 section, f32 u, f32 theta, f32 radius, i32 numControls) {
	VertexBinding binding = {};
	binding.section = section;
	binding.u = u;
	binding.theta = theta;
	binding.radius = radius;

	if (u <= 1.0e-5f || numControls <= 1) {
		binding.numInfluences = 1;
		binding.controls[0] = 0;
		binding.weights[0] = 1.0f;
		return binding;
	}
	if (u >= f32(numControls - 1) - 1.0e-5f) {
		binding.numInfluences = 1;
		binding.controls[0] = numControls - 1;
		binding.weights[0] = 1.0f;
		return binding;
	}

	const i32 segment = std::max(0, std::min(i32(std::floor(u)), numControls - 2));
	const f32 t = std::max(0.0f, std::min(1.0f, u - f32(segment)));
	const f32 t2 = t*t;
	const f32 t3 = t2*t;
	const f32 w0 = -0.5f*t + t2 - 0.5f*t3;
	const f32 w1 = 1.0f - 2.5f*t2 + 1.5f*t3;
	const f32 w2 = 0.5f*t + 2.0f*t2 - 1.5f*t3;
	const f32 w3 = -0.5f*t2 + 0.5f*t3;

	addBindingWeight(&binding, segment - 1, w0, numControls);
	addBindingWeight(&binding, segment, w1, numControls);
	addBindingWeight(&binding, segment + 1, w2, numControls);
	addBindingWeight(&binding, segment + 2, w3, numControls);

	f32 weightSum = 0.0f;
	for (i32 i = 0; i < binding.numInfluences; ++i) {
		weightSum += binding.weights[i];
	}
	if (binding.numInfluences == 0 || weightSum <= 0.0f) {
		i32 c = i32(std::round(u));
		c = std::max(0, std::min(c, numControls - 1));
		binding.numInfluences = 1;
		binding.controls[0] = c;
		binding.weights[0] = 1.0f;
		return binding;
	}

	for (i32 i = 0; i < binding.numInfluences; ++i) {
		binding.weights[i] /= weightSum;
	}
	return binding;
}

static vec3 weightedControlPosition(const std::vector<ControlPoint> &controls, const VertexBinding &binding) {
	vec3 p(0.0f);
	for (i32 i = 0; i < binding.numInfluences; ++i) {
		p += binding.weights[i]*controls[size_t(binding.controls[i])].position;
	}
	return p;
}

static vec3 safeTubeFrameRight(vec3 tangent) {
	vec3 seed(0.0f, 0.0f, 1.0f);
	if (std::fabs(dot(seed, tangent)) > 0.85f) {
		seed = vec3(0.0f, 1.0f, 0.0f);
	}

	vec3 right = cross(seed, tangent);
	if (sqNorm(right) < 1.0e-10f) {
		right = vec3(0.0f, 1.0f, 0.0f);
	}
	return normal(right);
}

static void updateTentaclePositions(TentacleState *state) {
	if (state == nullptr || state->sections <= 0 || state->radialSegments <= 0) {
		return;
	}

	std::vector<vec3> centers(size_t(state->sections), vec3(0.0f));
	for (i32 section = 0; section < state->sections; ++section) {
		const i32 vertexIdx = section*state->radialSegments;
		centers[size_t(section)] = weightedControlPosition(state->controls, state->bindings[size_t(vertexIdx)]);
	}

	std::vector<vec3> tangents(size_t(state->sections), vec3(1.0f, 0.0f, 0.0f));
	for (i32 section = 0; section < state->sections; ++section) {
		const i32 prevSection = std::max(0, section - 1);
		const i32 nextSection = std::min(state->sections - 1, section + 1);
		vec3 tangent = centers[size_t(nextSection)] - centers[size_t(prevSection)];
		if (sqNorm(tangent) < 1.0e-10f) {
			tangent = vec3(1.0f, 0.0f, 0.0f);
		}
		tangents[size_t(section)] = normal(tangent);
	}

	std::vector<vec3> rights(size_t(state->sections), vec3(0.0f, 1.0f, 0.0f));
	std::vector<vec3> ups(size_t(state->sections), vec3(0.0f, 0.0f, 1.0f));
	rights[0] = safeTubeFrameRight(tangents[0]);
	ups[0] = normal(cross(tangents[0], rights[0]));
	for (i32 section = 1; section < state->sections; ++section) {
		vec3 right = rights[size_t(section - 1)];
		right = right - dot(right, tangents[size_t(section)])*tangents[size_t(section)];
		if (sqNorm(right) < 1.0e-10f) {
			right = safeTubeFrameRight(tangents[size_t(section)]);
		} else {
			right = normal(right);
		}

		rights[size_t(section)] = right;
		ups[size_t(section)] = normal(cross(tangents[size_t(section)], right));
	}

	for (size_t i = 0; i < state->bindings.size(); ++i) {
		const VertexBinding binding = state->bindings[i];
		const i32 section = std::max(0, std::min(state->sections - 1, binding.section));
		const vec3 radial =
			sprCosf(binding.theta)*rights[size_t(section)] +
			sprSinf(binding.theta)*ups[size_t(section)];
		state->p[i] = centers[size_t(section)] + binding.radius*radial;
	}
}

static geom::Mesh makeTentacleMesh(
	TentacleState *stateOut,
	i32 numControls,
	i32 sections,
	i32 radialSegments) {
	TentacleState state = {};
	state.sections = sections;
	state.radialSegments = radialSegments;
	state.controls.resize(size_t(numControls));
	for (i32 i = 0; i < numControls; ++i) {
		const vec3 guide = guideControlPosition(i, numControls);
		state.controls[size_t(i)].guide = guide;
		state.controls[size_t(i)].position = guide;
		state.controls[size_t(i)].isPinned = i == 0 || i + 1 == numControls;
	}

	geom::Mesh mesh;
	mesh.positions.resize(size_t(sections)*size_t(radialSegments));
	state.bindings.resize(mesh.positions.size());

	for (i32 section = 0; section < sections; ++section) {
		const f32 s = f32(section)/f32(sections - 1);
		const f32 u = s*f32(numControls - 1);
		const f32 radius = 0.125f*(1.0f - 0.30f*s);

		for (i32 r = 0; r < radialSegments; ++r) {
			const f32 theta = 2.0f*f32(SPR_PI)*f32(r)/f32(radialSegments);
			const i32 vertexIdx = section*radialSegments + r;
			state.bindings[size_t(vertexIdx)] = makeVertexBinding(section, u, theta, radius, numControls);
		}
	}

	for (i32 section = 0; section + 1 < sections; ++section) {
		for (i32 r = 0; r < radialSegments; ++r) {
			const geom::vid v00 = section*radialSegments + r;
			const geom::vid v01 = section*radialSegments + ((r + 1)%radialSegments);
			const geom::vid v10 = (section + 1)*radialSegments + r;
			const geom::vid v11 = (section + 1)*radialSegments + ((r + 1)%radialSegments);
			geom::appendTriangle(&mesh, v00, v11, v10);
			geom::appendTriangle(&mesh, v00, v01, v11);
		}
	}

	state.p.resize(mesh.positions.size());
	updateTentaclePositions(&state);
	mesh.positions = state.p;
	geom::ensureMeshNormals(&mesh);

	state.p0 = state.p;
	state.q.assign(state.p.size(), quat::one());
	geom::buildMeshTopology(&state.topology, &mesh);
	geom::computeVertexAreas(&state.vertexAreas, &mesh);

	state.controlMass.assign(size_t(numControls), 1.0e-4f);
	for (size_t i = 0; i < state.bindings.size(); ++i) {
		const VertexBinding binding = state.bindings[i];
		const f32 area = state.vertexAreas[i];
		for (i32 j = 0; j < binding.numInfluences; ++j) {
			state.controlMass[size_t(binding.controls[j])] += std::fabs(binding.weights[j])*area;
		}
	}

	*stateOut = std::move(state);
	return mesh;
}

static EnergyReport evaluateEnergyAndGradient(
	const TentacleState &state,
	const ExperimentParams &params,
	const std::vector<ObstacleSample> &samples,
	std::vector<vec3> *controlGradOut,
	std::vector<vec3> *gradQOut) {
	const i32 n = i32(state.p.size());
	const bool needsGradient = controlGradOut != nullptr && gradQOut != nullptr;

	std::vector<vec3> gradP;
	if (needsGradient) {
		gradP.assign(size_t(n), vec3(0.0f));
		controlGradOut->assign(state.controls.size(), vec3(0.0f));
		gradQOut->assign(size_t(n), vec3(0.0f));
	}

	EnergyReport report = {};
	f64 weightedViolation = 0.0;
	f64 totalArea = 0.0;

	for (i32 i = 0; i < n; ++i) {
		const f32 area = state.vertexAreas[size_t(i)];
		const ObstacleSample sample = samples[size_t(i)];
		const f32 t = params.desiredClearance - sample.phi;
		const f32 violation = sprMax<f32>(0.0f, t);
		const f32 rho = stableSoftplus(t, params.softplusEps);
		const f32 rhoPrime = stableSigmoid(t, params.softplusEps);

		report.minPhi = sprMin<f32>(report.minPhi, sample.phi);
		report.maxViolation = sprMax<f32>(report.maxViolation, violation);
		if (violation > 0.0f) {
			++report.numViolating;
		}

		weightedViolation += f64(area)*f64(violation);
		totalArea += f64(area);
		report.clearance += 0.5*double(params.tempClear)*double(area)*double(rho)*double(rho);

		if (needsGradient) {
			gradP[size_t(i)] += -params.tempClear*area*rho*rhoPrime*sample.grad;
		}
	}

	for (i32 i = 0; i < n; ++i) {
		const i32 begin = state.topology.neighborOffsets[size_t(i)];
		const i32 end = state.topology.neighborOffsets[size_t(i) + 1];
		const i32 degree = end - begin;
		if (degree <= 0) {
			continue;
		}

		const rot3 R = rot3FromUnitQuatUnchecked(state.q[size_t(i)]);
		const rot3 Rinv = inverse(R);
		const f32 alphaBase = params.tempArap*state.vertexAreas[size_t(i)]/f32(degree);

		for (i32 k = begin; k < end; ++k) {
			const i32 j = state.topology.neighbors[size_t(k)];
			const vec3 e0 = state.p0[size_t(i)] - state.p0[size_t(j)];
			const vec3 e = state.p[size_t(i)] - state.p[size_t(j)];
			const vec3 r = R*e0 - e;

			report.arap += 0.5*double(alphaBase)*double(sqNorm(r));
			if (needsGradient) {
				gradP[size_t(i)] += -alphaBase*r;
				gradP[size_t(j)] += alphaBase*r;
				const vec3 localE = Rinv*e;
				(*gradQOut)[size_t(i)] += -2.0f*alphaBase*cross(e0, localE);
			}
		}
	}

	f64 guide2 = 0.0;
	i32 numFreeControls = 0;
	for (size_t c = 0; c < state.controls.size(); ++c) {
		const ControlPoint control = state.controls[c];
		const vec3 residual = control.position - control.guide;
		if (!control.isPinned) {
			++numFreeControls;
		}

		report.guide += 0.5*double(params.tempGuide)*double(sqNorm(residual));
		guide2 += double(sqNorm(residual));
		report.maxControlDisplacement = sprMax<f32>(report.maxControlDisplacement, length(residual));

		if (needsGradient) {
			(*controlGradOut)[c] += params.tempGuide*residual;
		}
	}

	for (i32 c = 1; c + 1 < i32(state.controls.size()); ++c) {
		const vec3 bend =
			state.controls[size_t(c - 1)].position -
			2.0f*state.controls[size_t(c)].position +
			state.controls[size_t(c + 1)].position;
		const vec3 guideBend =
			state.controls[size_t(c - 1)].guide -
			2.0f*state.controls[size_t(c)].guide +
			state.controls[size_t(c + 1)].guide;
		const vec3 residual = bend - guideBend;

		report.controlBend += 0.5*double(params.tempControlBend)*double(sqNorm(residual));
		if (needsGradient) {
			const vec3 grad = params.tempControlBend*residual;
			(*controlGradOut)[size_t(c - 1)] += grad;
			(*controlGradOut)[size_t(c)] += -2.0f*grad;
			(*controlGradOut)[size_t(c + 1)] += grad;
		}
	}

	if (needsGradient) {
		for (size_t i = 0; i < state.bindings.size(); ++i) {
			const VertexBinding binding = state.bindings[i];
			for (i32 j = 0; j < binding.numInfluences; ++j) {
				(*controlGradOut)[size_t(binding.controls[j])] += binding.weights[j]*gradP[i];
			}
		}

		for (size_t c = 0; c < state.controls.size(); ++c) {
			if (state.controls[c].isPinned) {
				(*controlGradOut)[c] = vec3(0.0f);
			} else {
				(*controlGradOut)[c].x = 0.0f;
			}
		}
	}

	report.total = report.clearance + report.arap + report.guide + report.controlBend;
	if (totalArea > 0.0) {
		report.avgViolation = f32(weightedViolation/totalArea);
	}
	if (!state.controls.empty()) {
		report.rmsGuide = f32(std::sqrt(guide2/double(state.controls.size())));
	}
	(void)numFreeControls;
	return report;
}

static void applyStep(
	TentacleState *trial,
	const TentacleState &state,
	const std::vector<vec3> &controlGrad,
	const std::vector<vec3> &gradQ,
	f32 controlStep,
	f32 rotationStep) {
	*trial = state;

	for (size_t c = 0; c < state.controls.size(); ++c) {
		if (state.controls[c].isPinned) {
			continue;
		}

		const f32 mass = state.controlMass[c] > 1.0e-8f ? state.controlMass[c] : 1.0f;
		vec3 step = (controlStep/mass)*controlGrad[c];
		step.x = 0.0f;
		trial->controls[c].position = state.controls[c].position - step;
	}

	updateTentaclePositions(trial);

	for (size_t i = 0; i < state.q.size(); ++i) {
		const f32 mass = state.vertexAreas[i] > 1.0e-8f ? state.vertexAreas[i] : 1.0f;
		trial->q[i] = state.q[i]*exp_t((-rotationStep/mass)*gradQ[i]);
		normalize(trial->q[i]);
	}
}

static const std::vector<ObstacleSample> &sampleObstacleField(
	ObstacleSampler *sampler,
	std::vector<ObstacleSample> *samples,
	ObstacleField obstacle,
	const TentacleState &state) {
	return adsdf::sampleAdsdfPointsBlocking(
		samples,
		sampler,
		obstacle.field,
		state.p,
		0.003f);
}

static EnergyReport evaluateWithCurrentSamples(
	ObstacleSampler *sampler,
	std::vector<ObstacleSample> *samples,
	ObstacleField obstacle,
	const TentacleState &state,
	const ExperimentParams &params,
	std::vector<vec3> *controlGrad,
	std::vector<vec3> *gradQ) {
	const std::vector<ObstacleSample> &currentSamples = sampleObstacleField(
		sampler,
		samples,
		obstacle,
		state);
	return evaluateEnergyAndGradient(state, params, currentSamples, controlGrad, gradQ);
}

static void collectFreeControls(
	const TentacleState &state,
	std::vector<i32> *freeControlsOut,
	std::vector<i32> *controlToVarOut) {
	freeControlsOut->clear();
	controlToVarOut->assign(state.controls.size(), -1);

	for (i32 c = 0; c < i32(state.controls.size()); ++c) {
		if (state.controls[size_t(c)].isPinned) {
			continue;
		}

		(*controlToVarOut)[size_t(c)] = i32(freeControlsOut->size());
		freeControlsOut->push_back(c);
	}
}

static f64 buildControlLmSystem(
	const TentacleState &state,
	const ExperimentParams &params,
	const std::vector<ObstacleSample> &samples,
	const std::vector<i32> &controlToVar,
	std::vector<f64> *jtJOut,
	std::vector<f64> *jtROut) {
	const i32 numVars = 2;
	i32 maxVar = -1;
	for (i32 idx : controlToVar) {
		maxVar = std::max(maxVar, idx);
	}
	const i32 n = numVars*(maxVar + 1);

	jtJOut->assign(size_t(n*n), 0.0);
	jtROut->assign(size_t(n), 0.0);

	std::vector<f64> j(size_t(n), 0.0);
	f64 residual2 = 0.0;

	for (i32 i = 0; i < i32(state.p.size()); ++i) {
		std::fill(j.begin(), j.end(), 0.0);

		const f32 area = state.vertexAreas[size_t(i)];
		const ObstacleSample sample = samples[size_t(i)];
		const f32 t = params.desiredClearance - sample.phi;
		const f32 rho = stableSoftplus(t, params.softplusEps);
		const f32 rhoPrime = stableSigmoid(t, params.softplusEps);
		const f64 scale = std::sqrt(double(params.tempClear)*double(area));
		const f64 r = scale*double(rho);
		const f64 dCommon = -scale*double(rhoPrime);
		const VertexBinding binding = state.bindings[size_t(i)];

		for (i32 influence = 0; influence < binding.numInfluences; ++influence) {
			const i32 var = controlToVar[size_t(binding.controls[influence])];
			if (var < 0) {
				continue;
			}

			const f64 w = double(binding.weights[influence]);
			j[size_t(2*var + 0)] += dCommon*w*double(sample.grad.y);
			j[size_t(2*var + 1)] += dCommon*w*double(sample.grad.z);
		}

		numeric::accumulateDenseResidual(jtJOut, jtROut, &residual2, j, r);
	}

	const f64 guideScale = std::sqrt(double(params.tempGuide));
	for (i32 c = 0; c < i32(state.controls.size()); ++c) {
		const i32 var = controlToVar[size_t(c)];
		if (var < 0) {
			continue;
		}

		std::fill(j.begin(), j.end(), 0.0);
		j[size_t(2*var + 0)] = guideScale;
		numeric::accumulateDenseResidual(
			jtJOut,
			jtROut,
			&residual2,
			j,
			guideScale*double(state.controls[size_t(c)].position.y - state.controls[size_t(c)].guide.y));

		std::fill(j.begin(), j.end(), 0.0);
		j[size_t(2*var + 1)] = guideScale;
		numeric::accumulateDenseResidual(
			jtJOut,
			jtROut,
			&residual2,
			j,
			guideScale*double(state.controls[size_t(c)].position.z - state.controls[size_t(c)].guide.z));
	}

	const f64 bendScale = std::sqrt(double(params.tempControlBend));
	for (i32 c = 1; c + 1 < i32(state.controls.size()); ++c) {
		const i32 controlIds[3] = { c - 1, c, c + 1 };
		const f64 coeffs[3] = { 1.0, -2.0, 1.0 };
		const vec3 bend =
			state.controls[size_t(c - 1)].position -
			2.0f*state.controls[size_t(c)].position +
			state.controls[size_t(c + 1)].position;
		const vec3 guideBend =
			state.controls[size_t(c - 1)].guide -
			2.0f*state.controls[size_t(c)].guide +
			state.controls[size_t(c + 1)].guide;
		const vec3 residual = bend - guideBend;

		for (i32 componentIdx = 0; componentIdx < 2; ++componentIdx) {
			std::fill(j.begin(), j.end(), 0.0);
			for (i32 k = 0; k < 3; ++k) {
				const i32 var = controlToVar[size_t(controlIds[k])];
				if (var < 0) {
					continue;
				}
				j[size_t(2*var + componentIdx)] += bendScale*coeffs[k];
			}

			const f64 r = bendScale*double(componentIdx == 0 ? residual.y : residual.z);
			numeric::accumulateDenseResidual(jtJOut, jtROut, &residual2, j, r);
		}
	}

	return 0.5*residual2;
}

static f64 buildControlLmSystemWithCurrentSamples(
	ObstacleSampler *sampler,
	std::vector<ObstacleSample> *samples,
	ObstacleField obstacle,
	const TentacleState &state,
	const ExperimentParams &params,
	const std::vector<i32> &controlToVar,
	std::vector<f64> *jtJOut,
	std::vector<f64> *jtROut) {
	const std::vector<ObstacleSample> &currentSamples = sampleObstacleField(
		sampler,
		samples,
		obstacle,
		state);
	return buildControlLmSystem(state, params, currentSamples, controlToVar, jtJOut, jtROut);
}

static f32 applyControlLmDelta(
	TentacleState *trial,
	const TentacleState &state,
	const std::vector<i32> &freeControls,
	const std::vector<f64> &delta) {
	*trial = state;
	f32 maxDelta = 0.0f;

	for (i32 var = 0; var < i32(freeControls.size()); ++var) {
		const i32 c = freeControls[size_t(var)];
		const vec3 step(
			0.0f,
			f32(delta[size_t(2*var + 0)]),
			f32(delta[size_t(2*var + 1)]));
		trial->controls[size_t(c)].position += step;
		maxDelta = sprMax<f32>(maxDelta, length(step));
	}

	updateTentaclePositions(trial);
	return maxDelta;
}

static ControlLmStats refineControlsLm(
	TentacleState *state,
	const ExperimentParams &params,
	ObstacleSampler *sampler,
	std::vector<ObstacleSample> *samples,
	ObstacleField obstacle,
	expdiag::RunDiagnostics *run) {
	ControlLmStats stats = {};
	std::vector<i32> freeControls;
	std::vector<i32> controlToVar;
	collectFreeControls(*state, &freeControls, &controlToVar);
	const i32 n = 2*i32(freeControls.size());
	if (n <= 0) {
		return stats;
	}

	EnergyReport current = evaluateWithCurrentSamples(
		sampler,
		samples,
		obstacle,
		*state,
		params,
		nullptr,
		nullptr);
	stats.beforeEnergy = current.total;

	std::vector<f64> jtJ;
	std::vector<f64> jtR;
	stats.beforeControlObjective = buildControlLmSystemWithCurrentSamples(
		sampler,
		samples,
		obstacle,
		*state,
		params,
		controlToVar,
		&jtJ,
		&jtR);

	f64 lambda = 1.0e-3;
	numeric::DenseLmStepParams lmStepParams = {};
	lmStepParams.minLambda = 1.0e-8;
	lmStepParams.lambdaDecrease = 0.35;
	lmStepParams.lambdaIncrease = 8.0;
	lmStepParams.blockSize = 2;
	lmStepParams.maxBlockNorm = 0.35;
	lmStepParams.damping.minDiag = 1.0e-8;
	lmStepParams.damping.pivotEps = 1.0e-14;
	std::printf(
		"control LM start: energy=%g controlObjective=%g maxViolation=%g rmsGuide=%g vars=%d\n",
		current.total,
		stats.beforeControlObjective,
		current.maxViolation,
		current.rmsGuide,
		n);

	for (i32 iter = 0; iter < 18; ++iter) {
		const f64 controlObjective = buildControlLmSystemWithCurrentSamples(
			sampler,
			samples,
			obstacle,
			*state,
			params,
			controlToVar,
			&jtJ,
			&jtR);

		bool accepted = false;
		bool shouldStop = false;
		for (i32 attempt = 0; attempt < 10; ++attempt) {
			std::vector<f64> delta;
			numeric::DenseLmStepResult lmStep = {};
			lmStepParams.lambda = lambda;
			if (!numeric::solveDenseLmStep(&delta, &lmStep, jtJ, jtR, n, lmStepParams)) {
				numeric::increaseDenseLmLambda(&lambda, lmStepParams);
				continue;
			}

			TentacleState trial;
			const f32 appliedMaxDelta = applyControlLmDelta(&trial, *state, freeControls, delta);
			const EnergyReport trialReport = evaluateWithCurrentSamples(
				sampler,
				samples,
				obstacle,
				trial,
				params,
				nullptr,
				nullptr);

			const bool violationOk = trialReport.maxViolation <= sprMax<f32>(current.maxViolation, 1.0e-5f);
			const f64 energyGain = current.total - trialReport.total;
			if (trialReport.total < current.total && violationOk) {
				*state = std::move(trial);
				current = trialReport;
				expdiag::appendSample(run, makeDiagnosticSample(expdiag::nextIteration(*run), current));
				numeric::decreaseDenseLmLambda(&lambda, lmStepParams);
				++stats.accepted;
				accepted = true;
				std::printf(
					"control LM iter %2d accepted energy=%10.6f controlObjective=%10.6f maxViolation=% .6f rmsGuide=% .6f maxDelta=% .6f lambda=%g\n",
					iter,
					current.total,
					controlObjective,
					current.maxViolation,
					current.rmsGuide,
					appliedMaxDelta,
					lambda);
				shouldStop = appliedMaxDelta < 1.0e-3f || energyGain < 1.0e-8;
				break;
			}

			numeric::increaseDenseLmLambda(&lambda, lmStepParams);
		}

		++stats.iterations;
		if (!accepted) {
			++stats.rejected;
			std::printf("control LM stopped at iter %d after rejecting trial steps\n", iter);
			break;
		}
		if (shouldStop) {
			std::printf("control LM converged at iter %d\n", iter);
			break;
		}
	}

	stats.afterEnergy = current.total;
	stats.afterControlObjective = buildControlLmSystemWithCurrentSamples(
		sampler,
		samples,
		obstacle,
		*state,
		params,
		controlToVar,
		&jtJ,
		&jtR);
	std::printf(
		"control LM done: accepted=%d energy=%g -> %g controlObjective=%g -> %g\n",
		stats.accepted,
		stats.beforeEnergy,
		stats.afterEnergy,
		stats.beforeControlObjective,
		stats.afterControlObjective);

	return stats;
}

static void optimize(
	TentacleState *state,
	const ExperimentParams &params,
	ObstacleSampler *sampler,
	std::vector<ObstacleSample> *samples,
	ObstacleField obstacle,
	expdiag::RunDiagnostics *run) {
	std::vector<vec3> controlGrad;
	std::vector<vec3> gradQ;

	f32 controlStep = params.controlStep;
	f32 rotationStep = params.rotationStep;

	for (i32 iter = 0; iter < params.iterations; ++iter) {
		const EnergyReport current = evaluateWithCurrentSamples(
			sampler,
			samples,
			obstacle,
			*state,
			params,
			&controlGrad,
			&gradQ);

		if (iter % 20 == 0 || iter + 1 == params.iterations) {
			std::printf(
				"iter %3d energy=%10.6f maxViolation=% .6f avgViolation=% .6f rmsGuide=% .6f maxCtrl=% .6f\n",
				iter,
				current.total,
				current.maxViolation,
				current.avgViolation,
				current.rmsGuide,
				current.maxControlDisplacement);
		}

		bool accepted = false;
		bool shouldStop = false;
		for (i32 attempt = 0; attempt < 12; ++attempt) {
			TentacleState trial;
			applyStep(&trial, *state, controlGrad, gradQ, controlStep, rotationStep);
			const EnergyReport trialReport = evaluateWithCurrentSamples(
				sampler,
				samples,
				obstacle,
				trial,
				params,
				nullptr,
				nullptr);

			if (trialReport.total <= current.total) {
				*state = std::move(trial);
				expdiag::appendSample(run, makeDiagnosticSample(expdiag::nextIteration(*run), trialReport));
				shouldStop = expdiag::recordStopIfReached(run, params.stop);
				controlStep = sprMin<f32>(controlStep*1.025f, params.controlStep);
				rotationStep = sprMin<f32>(rotationStep*1.025f, params.rotationStep);
				accepted = true;
				break;
			}

			controlStep *= 0.5f;
			rotationStep *= 0.5f;
		}

		if (!accepted) {
			std::printf("line search failed at iteration %d\n", iter);
			break;
		}
		if (shouldStop) {
			break;
		}
	}
}

int main() {
	expdiag::RunDiagnostics run;
	expdiag::beginRun(&run, TENTACLE_REPORT_LABEL);

	geom::Mesh obstacle;
	makeObstacleMesh(&obstacle);

	geom::Bvh obstacleBvh;
	geom::buildBvh(&obstacleBvh, &obstacle);
	if (obstacleBvh.root < 0) {
		std::fprintf(stderr, "failed to build obstacle BVH\n");
		return 1;
	}

	const i32 numControls = 9;
	const i32 sections = 73;
	const i32 radialSegments = 36;

	TentacleState state;
	geom::Mesh tentacle = makeTentacleMesh(&state, numControls, sections, radialSegments);
	const TentacleState initialState = state;
	const std::vector<ControlPoint> guideControls = state.controls;

	ExperimentParams params = {};
	geom::GpuMesh obstacleMesh_d = {};
	geom::GpuBvh obstacleBvh_d = {};
	ObstacleSampler sampler = {};
	geom::initGpuMesh(&obstacleMesh_d);
	geom::initGpuBvh(&obstacleBvh_d);
	adsdf::initGpuAdsdfSampler(&sampler);
	geom::uploadGpuMesh(&obstacleMesh_d, &obstacle);
	geom::uploadGpuBvh(&obstacleBvh_d, &obstacleBvh);
	std::vector<ObstacleSample> samples;

	const geom::Aabb obstacleBounds = geom::computeMeshBounds(&obstacle);
	const vec3 obstacleExtent = geom::aabbExtent(obstacleBounds);
	const f32 maxObstacleExtent = sprMax<f32>(obstacleExtent.x, sprMax<f32>(obstacleExtent.y, obstacleExtent.z));
	const vec3 adsdfPadding(0.35f*maxObstacleExtent);

	adsdf::AdsdfDesc desc;
	adsdf::initDesc(
		&desc,
		48,
		40,
		40,
		2,
		obstacleBounds.lower - adsdfPadding,
		obstacleBounds.upper + adsdfPadding,
		adsdf::AdsdfFilterKind::Linear);

	adsdf::AdsdfLinearGrid linearGrid = {};
	adsdf::allocLinearGrid(&linearGrid, &desc);

	adsdf::MeshAdsdfBuildParams buildParams = adsdf::defaultMeshAdsdfBuildParams();
	buildParams.lsq.fineRadius = 2;
	buildParams.lsq.fineExtent = 0.55f;
	buildParams.lsq.regularization = 1.0e-7f;
	buildParams.nearSurfaceEps = 1.0e-4f;
	buildParams.signMethod = geom::MeshSignMethod::RayParity;
	adsdf::buildMeshLsqBlocking(&linearGrid, &obstacleMesh_d, &obstacleBvh_d, buildParams);

	adsdf::AdsdfTextureGrid textureGrid = {};
	adsdf::allocTextureGrid(&textureGrid, &desc);
	adsdf::uploadTextureGrid(&textureGrid, &linearGrid);

	auto cleanupCuda = [&]() {
		adsdf::freeGpuAdsdfSampler(&sampler);
		adsdf::freeTextureGrid(&textureGrid);
		adsdf::freeLinearGrid(&linearGrid);
		geom::freeGpuBvh(&obstacleBvh_d);
		geom::freeGpuMesh(&obstacleMesh_d);
	};

	ObstacleField obstacleField = {};
	obstacleField.field = adsdf::makeView(&textureGrid);

	const EnergyReport before = evaluateWithCurrentSamples(
		&sampler,
		&samples,
		obstacleField,
		state,
		params,
		nullptr,
		nullptr);
	expdiag::appendSample(&run, makeDiagnosticSample(0, before));

	std::printf(
		"before " TENTACLE_REPORT_LABEL ": energy=%g maxViolation=%g avgViolation=%g rmsGuide=%g violating=%d/%zu controls=%zu\n",
		before.total,
		before.maxViolation,
		before.avgViolation,
		before.rmsGuide,
		before.numViolating,
		state.p.size(),
		state.controls.size());

	optimize(&state, params, &sampler, &samples, obstacleField, &run);
	const ControlLmStats lmStats = refineControlsLm(&state, params, &sampler, &samples, obstacleField, &run);
	(void)lmStats;

	const EnergyReport after = evaluateWithCurrentSamples(
		&sampler,
		&samples,
		obstacleField,
		state,
		params,
		nullptr,
		nullptr);

	std::printf(
		"after " TENTACLE_REPORT_LABEL ":  energy=%g maxViolation=%g avgViolation=%g rmsGuide=%g maxCtrl=%g violating=%d/%zu\n",
		after.total,
		after.maxViolation,
		after.avgViolation,
		after.rmsGuide,
		after.maxControlDisplacement,
		after.numViolating,
		state.p.size());

	const OrientationReport initialOrient = evaluateTubeOrientation(initialState, tentacle.triangles);
	const OrientationReport afterOrient = evaluateTubeOrientation(state, tentacle.triangles);
	std::printf(
		"tentacle normal check: initial inward=%d/%d minDot=%g avgDot=%g; optimized inward=%d/%d minDot=%g avgDot=%g\n",
		initialOrient.numInward,
		initialOrient.numFaces,
		initialOrient.minDot,
		initialOrient.avgDot,
		afterOrient.numInward,
		afterOrient.numFaces,
		afterOrient.minDot,
		afterOrient.avgDot);

	if (after.total >= before.total || after.maxViolation >= before.maxViolation || after.avgViolation >= before.avgViolation) {
		std::fprintf(stderr, TENTACLE_REPORT_LABEL " armature experiment did not improve contact\n");
		cleanupCuda();
		return 1;
	}
	if (initialOrient.numInward > 0 || afterOrient.numInward > 0) {
		std::fprintf(stderr, TENTACLE_REPORT_LABEL " armature experiment produced inward-facing tube normals\n");
		cleanupCuda();
		return 1;
	}

	writeObj("../working/" TENTACLE_OUTPUT_PREFIX "_obstacle.obj", obstacle);
	writeObj("../working/" TENTACLE_OUTPUT_PREFIX "_initial.obj", tentacle);
	writeObj("../working/" TENTACLE_OUTPUT_PREFIX "_optimized.obj", state.p, tentacle.triangles);
	writeControlObj("../working/" TENTACLE_OUTPUT_PREFIX "_guide_controls.obj", guideControls, true);
	writeControlObj("../working/" TENTACLE_OUTPUT_PREFIX "_optimized_controls.obj", state.controls, false);

	const geom::Mesh guideControlMesh = makeControlCurveMesh(guideControls, true, 0.026f, 12);
	const geom::Mesh optimizedControlMesh = makeControlCurveMesh(state.controls, false, 0.026f, 12);

	geom::MeshPreviewObject initialPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&obstacle, vec3(0.58f, 0.62f, 0.66f), false),
		geom::makeMeshPreviewObject(tentacle.positions, tentacle.triangles, vec3(0.95f, 0.50f, 0.18f), false),
	};
	geom::MeshPreviewParams previewParams = geom::defaultMeshPreviewParams(1000, 650);
	const geom::Aabb initialPreviewBounds = geom::computePreviewBounds(initialPreviewObjects, 2);
	previewParams.camera = geom::makeOrbitPreviewCamera(
		initialPreviewBounds,
		vec3(2.5f, -2.7f, 1.2f));
	if (!geom::renderMeshPreviewPpm("../working/" TENTACLE_OUTPUT_PREFIX "_initial.ppm", initialPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_initial.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject optimizedPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&obstacle, vec3(0.58f, 0.62f, 0.66f), false),
		geom::makeMeshPreviewObject(state.p, tentacle.triangles, vec3(0.12f, 0.62f, 0.78f), false),
	};
	const geom::Aabb optimizedPreviewBounds = geom::computePreviewBounds(optimizedPreviewObjects, 2);
	previewParams.camera = geom::makeOrbitPreviewCamera(
		optimizedPreviewBounds,
		vec3(2.5f, -2.7f, 1.2f));
	if (!geom::renderMeshPreviewPpm("../working/" TENTACLE_OUTPUT_PREFIX "_optimized.ppm", optimizedPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_optimized.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject optimizedTubeOnlyObjects[1] = {
		geom::makeMeshPreviewObject(state.p, tentacle.triangles, vec3(0.12f, 0.62f, 0.78f), false),
	};
	const geom::Aabb optimizedTubeOnlyBounds = geom::computePreviewBounds(optimizedTubeOnlyObjects, 1);
	previewParams.camera = geom::makeOrbitPreviewCamera(
		optimizedTubeOnlyBounds,
		vec3(2.5f, -2.7f, 1.2f));
	if (!geom::renderMeshPreviewPpm("../working/" TENTACLE_OUTPUT_PREFIX "_optimized_tube.ppm", optimizedTubeOnlyObjects, 1, previewParams)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_optimized_tube.ppm\n");
		cleanupCuda();
		return 1;
	}

	previewParams.camera = geom::makeOrbitPreviewCamera(
		optimizedPreviewBounds,
		vec3(2.5f, -2.7f, 1.2f));
	previewParams.shadeMode = geom::MeshPreviewShadeMode::Normal;
	if (!geom::renderMeshPreviewPpm("../working/" TENTACLE_OUTPUT_PREFIX "_optimized_normals.ppm", optimizedPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_optimized_normals.ppm\n");
		cleanupCuda();
		return 1;
	}

	previewParams.shadeMode = geom::MeshPreviewShadeMode::Position;
	previewParams.falseColorMin = optimizedPreviewBounds.lower;
	previewParams.falseColorMax = optimizedPreviewBounds.upper;
	if (!geom::renderMeshPreviewPpm("../working/" TENTACLE_OUTPUT_PREFIX "_optimized_position.ppm", optimizedPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_optimized_position.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject controlPreviewObjects[3] = {
		geom::makeMeshPreviewObject(&obstacle, vec3(0.58f, 0.62f, 0.66f), false),
		geom::makeMeshPreviewObject(&guideControlMesh, vec3(0.95f, 0.56f, 0.12f), false),
		geom::makeMeshPreviewObject(&optimizedControlMesh, vec3(0.08f, 0.76f, 0.88f), false),
	};
	const geom::Aabb controlPreviewBounds = geom::computePreviewBounds(controlPreviewObjects, 3);
	previewParams.shadeMode = geom::MeshPreviewShadeMode::LitColor;
	previewParams.camera = geom::makeOrbitPreviewCamera(
		controlPreviewBounds,
		vec3(2.5f, -2.7f, 1.2f));
	if (!geom::renderMeshPreviewPpm("../working/" TENTACLE_OUTPUT_PREFIX "_controls.ppm", controlPreviewObjects, 3, previewParams)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_controls.ppm\n");
		cleanupCuda();
		return 1;
	}

	const vec3 fieldPreviewCenter = geom::aabbCentroid(obstacleBounds);
	const f32 fieldPreviewRadius = 0.5f*length((obstacleBounds.upper + adsdfPadding) - (obstacleBounds.lower - adsdfPadding));
	if (!renderAdsdfPreviewPpm(
		"../working/" TENTACLE_OUTPUT_PREFIX "_field.ppm",
		obstacleField.field,
		fieldPreviewCenter,
		fieldPreviewRadius,
		vec3(2.5f, -2.7f, 1.2f),
		1000,
		650)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX "_field.ppm\n");
		cleanupCuda();
		return 1;
	}

	if (!expdiag::writeRunArtifacts("../working/" TENTACLE_OUTPUT_PREFIX, &run)) {
		std::fprintf(stderr, "failed to write " TENTACLE_OUTPUT_PREFIX " diagnostics\n");
		cleanupCuda();
		return 1;
	}

	cleanupCuda();
	std::printf("wrote ../working/" TENTACLE_OUTPUT_PREFIX "_{obstacle,initial,optimized,guide_controls,optimized_controls}.obj and preview/control/field diagnostic PPMs\n");
	return 0;
}
