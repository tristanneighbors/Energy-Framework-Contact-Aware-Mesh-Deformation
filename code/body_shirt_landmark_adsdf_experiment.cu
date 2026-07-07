#include "mesh_to_adsdf.cuh"
#include "dbg/cuda_utils.cuh"
#include "experiment_diagnostics.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

struct ExperimentParams {
	f32 desiredOffset = 0.025f;
	f32 softplusEps = 0.008f;

	f32 tempOffset = 0.0f;
	f32 tempClear = 5000.0f;
	f32 tempArap = 2.0f;
	f32 tempArapStart = 80.0f;
	f32 arapRelaxFactor = 0.5f;
	i32 arapStageMinIterations = 24;
	i32 arapStagePatience = 18;
	f64 arapStageRelativeProgress = 5.0e-4;
	f64 arapStageAbsoluteProgress = 1.0e-6;
	f32 tempLand = 1.2f;
	f32 tempCenter = 0.02f;

	i32 seedSmoothingIterations = 480;
	i32 iterations = 720;
	f32 positionStep = 0.004f;
	f32 rotationStep = 0.02f;
	expdiag::StopCriteria stop;
};

struct Landmark {
	i32 vertexIdx;
	vec3 target;
	f32 weight;
	i32 group;
};

struct ExperimentState {
	std::vector<vec3> p0;
	std::vector<vec3> p;
	std::vector<quat> q;
	std::vector<f32> vertexAreas;
	std::vector<f32> kappa;
	geom::MeshTopology topology;
	vec3 center0;
	f32 totalArea = 0.0f;
};

struct EnergyReport {
	f64 total = 0.0;
	f64 offset = 0.0;
	f64 clearance = 0.0;
	f64 arap = 0.0;
	f64 landmark = 0.0;
	f64 center = 0.0;

	f32 minPhi = F32_MAX;
	f32 maxViolation = 0.0f;
	f32 avgViolation = 0.0f;
	f32 rmsOffset = 0.0f;
	f32 rmsLandmark = 0.0f;
	i32 numViolating = 0;
};

struct ArapContinuationState {
	f32 weight = 0.0f;
	i32 stageStartIteration = 0;
	i32 lastProgressIteration = 0;
	f64 stageStartMetric = F64_MAX;
	f64 bestMetric = F64_MAX;
};

static expdiag::IterationSample makeDiagnosticSample(i32 iteration, const EnergyReport &report, f32 arapWeight) {
	expdiag::IterationSample sample = {};
	sample.iteration = iteration;
	sample.energy = report.total;
	sample.maxViolation = report.maxViolation;
	sample.avgViolation = report.avgViolation;
	sample.minPhi = report.minPhi;
	sample.rmsOffset = report.rmsOffset;
	sample.rmsLandmark = report.rmsLandmark;
	sample.maxControl = arapWeight;
	sample.numViolating = report.numViolating;
	return sample;
}

template<class RadiusFunc>
static void makeUvSphereMesh(
	geom::Mesh *meshOut,
	i32 rings,
	i32 segments,
	RadiusFunc radiusFunc) {
	geom::clearMesh(meshOut);

	auto dirFromAngles = [](f32 theta, f32 phi) {
		const f32 s = sprSinf(phi);
		return vec3(s*sprCosf(theta), s*sprSinf(theta), sprCosf(phi));
	};

	const vec3 topDir(0.0f, 0.0f, 1.0f);
	meshOut->positions.push_back(topDir*radiusFunc(0.0f, 0.0f, topDir));

	for (i32 r = 1; r < rings; ++r) {
		const f32 phi = f32(SPR_PI)*f32(r)/f32(rings);
		for (i32 s = 0; s < segments; ++s) {
			const f32 theta = 2.0f*f32(SPR_PI)*f32(s)/f32(segments);
			const vec3 dir = dirFromAngles(theta, phi);
			meshOut->positions.push_back(dir*radiusFunc(theta, phi, dir));
		}
	}

	const vec3 bottomDir(0.0f, 0.0f, -1.0f);
	const geom::vid bottomIdx = geom::vid(meshOut->positions.size());
	meshOut->positions.push_back(bottomDir*radiusFunc(0.0f, f32(SPR_PI), bottomDir));

	auto ringVertex = [segments](i32 r, i32 s) {
		const i32 wrapped = (s % segments + segments) % segments;
		return geom::vid(1 + (r - 1)*segments + wrapped);
	};

	for (i32 s = 0; s < segments; ++s) {
		geom::appendTriangle(meshOut, 0, ringVertex(1, s), ringVertex(1, s + 1));
	}

	for (i32 r = 1; r + 1 < rings; ++r) {
		for (i32 s = 0; s < segments; ++s) {
			const geom::vid v00 = ringVertex(r, s);
			const geom::vid v01 = ringVertex(r, s + 1);
			const geom::vid v10 = ringVertex(r + 1, s);
			const geom::vid v11 = ringVertex(r + 1, s + 1);
			geom::appendTriangle(meshOut, v00, v10, v11);
			geom::appendTriangle(meshOut, v00, v11, v01);
		}
	}

	for (i32 s = 0; s < segments; ++s) {
		geom::appendTriangle(meshOut, ringVertex(rings - 1, s), bottomIdx, ringVertex(rings - 1, s + 1));
	}

	geom::ensureMeshNormals(meshOut);
}

static bool writeObj(const char *path, const geom::Mesh &mesh) {
	return geom::tryWriteObj(path, mesh);
}

static bool writeLandmarksObj(const char *path, const std::vector<Landmark> &landmarks) {
	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	for (Landmark landmark : landmarks) {
		std::fprintf(
			file,
			"v %.9g %.9g %.9g\n",
			landmark.target.x,
			landmark.target.y,
			landmark.target.z);
	}

	std::fclose(file);
	return true;
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
		fromDirection = vec3(-2.8f, 0.35f, 1.1f);
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

static std::vector<i32> findBoundaryVertices(const geom::Mesh &mesh) {
	std::vector<std::pair<i32, i32>> edges;
	edges.reserve(mesh.triangles.size()*3);

	auto appendEdge = [&edges](i32 a, i32 b) {
		if (a > b) {
			std::swap(a, b);
		}
		edges.push_back(std::make_pair(a, b));
	};

	for (geom::MeshTri tri : mesh.triangles) {
		appendEdge(tri.v0, tri.v1);
		appendEdge(tri.v1, tri.v2);
		appendEdge(tri.v2, tri.v0);
	}

	std::sort(edges.begin(), edges.end());
	std::vector<char> isBoundary(mesh.positions.size(), 0);
	for (size_t i = 0; i < edges.size();) {
		size_t j = i + 1;
		while (j < edges.size() && edges[j] == edges[i]) {
			++j;
		}
		if (j - i == 1) {
			if (edges[i].first >= 0 && edges[i].first < i32(isBoundary.size())) {
				isBoundary[size_t(edges[i].first)] = 1;
			}
			if (edges[i].second >= 0 && edges[i].second < i32(isBoundary.size())) {
				isBoundary[size_t(edges[i].second)] = 1;
			}
		}
		i = j;
	}

	std::vector<i32> boundary;
	for (i32 i = 0; i < i32(isBoundary.size()); ++i) {
		if (isBoundary[size_t(i)] != 0) {
			boundary.push_back(i);
		}
	}
	return boundary;
}

static std::vector<Landmark> makeBoundaryLandmarks(const geom::Mesh &shirt, const geom::Mesh &posedShirt) {
	std::vector<i32> boundary = findBoundaryVertices(shirt);
	std::vector<Landmark> landmarks;
	landmarks.reserve(boundary.size());

	for (i32 vertexIdx : boundary) {
		Landmark landmark = {};
		landmark.vertexIdx = vertexIdx;
		landmark.target = posedShirt.positions[size_t(vertexIdx)];
		landmark.weight = 1.0f;
		landmark.group = 0;
		landmarks.push_back(landmark);
	}
	return landmarks;
}

static void applyHarmonicRetargetSeed(
	ExperimentState *state,
	const std::vector<Landmark> &landmarks,
	i32 iterations) {
	if (state == nullptr) {
		return;
	}

	std::vector<geom::vid> fixedVertices;
	std::vector<vec3> fixedTargets;
	fixedVertices.reserve(landmarks.size());
	fixedTargets.reserve(landmarks.size());
	for (Landmark landmark : landmarks) {
		fixedVertices.push_back(geom::vid(landmark.vertexIdx));
		fixedTargets.push_back(landmark.target);
	}
	geom::makeHarmonicVertexDisplacementSeed(
		&state->p,
		state->p0,
		state->topology,
		fixedVertices,
		fixedTargets,
		iterations);
}

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

static ArapContinuationState makeArapContinuation(const ExperimentParams &params) {
	ArapContinuationState state = {};
	state.weight = sprMax<f32>(params.tempArap, params.tempArapStart);
	return state;
}

static f64 arapContinuationMetric(const EnergyReport &report) {
	return report.clearance + report.landmark + report.offset;
}

static bool isFinalArapStage(const ArapContinuationState &continuation, const ExperimentParams &params) {
	const f32 tolerance = 1.0e-4f*sprMax<f32>(1.0f, params.tempArap);
	return continuation.weight <= params.tempArap + tolerance;
}

static bool maybeRelaxArapContinuation(
	ArapContinuationState *continuation,
	const ExperimentParams &params,
	i32 iteration,
	const EnergyReport &report) {
	if (continuation == nullptr) {
		return false;
	}

	const f64 metric = arapContinuationMetric(report);
	if (continuation->stageStartMetric == F64_MAX) {
		continuation->stageStartMetric = metric;
		continuation->bestMetric = metric;
		continuation->stageStartIteration = iteration;
		continuation->lastProgressIteration = iteration;
		return false;
	}

	const f64 progressThreshold = std::max(
		params.arapStageAbsoluteProgress,
		params.arapStageRelativeProgress*std::max(1.0, std::fabs(continuation->stageStartMetric)));
	if (metric + progressThreshold < continuation->bestMetric) {
		continuation->bestMetric = metric;
		continuation->lastProgressIteration = iteration;
		return false;
	}

	if (isFinalArapStage(*continuation, params)) {
		return false;
	}

	const i32 stageAge = iteration - continuation->stageStartIteration + 1;
	const i32 stalledIterations = iteration - continuation->lastProgressIteration;
	if (stageAge < params.arapStageMinIterations || stalledIterations < params.arapStagePatience) {
		return false;
	}

	const f32 oldWeight = continuation->weight;
	continuation->weight = sprMax<f32>(params.tempArap, continuation->weight*params.arapRelaxFactor);
	continuation->stageStartIteration = iteration + 1;
	continuation->lastProgressIteration = iteration + 1;
	continuation->stageStartMetric = F64_MAX;
	continuation->bestMetric = F64_MAX;
	std::printf(
		"relaxed ARAP at iteration %d: metric=%g weight %.3f -> %.3f after %d stalled iterations\n",
		iteration,
		metric,
		oldWeight,
		continuation->weight,
		stalledIterations);
	return true;
}

static void forceRelaxArapContinuation(
	ArapContinuationState *continuation,
	const ExperimentParams &params,
	i32 nextIteration,
	const char *reason) {
	if (continuation == nullptr || isFinalArapStage(*continuation, params)) {
		return;
	}

	const f32 oldWeight = continuation->weight;
	continuation->weight = sprMax<f32>(params.tempArap, continuation->weight*params.arapRelaxFactor);
	continuation->stageStartIteration = nextIteration;
	continuation->lastProgressIteration = nextIteration;
	continuation->stageStartMetric = F64_MAX;
	continuation->bestMetric = F64_MAX;
	std::printf(
		"relaxed ARAP at iteration %d after %s: weight %.3f -> %.3f\n",
		nextIteration,
		reason != nullptr ? reason : "stall",
		oldWeight,
		continuation->weight);
}

static vec3 areaWeightedCenter(const std::vector<vec3> &positions, const std::vector<f32> &areas, f32 totalArea) {
	vec3 center(0.0f);
	if (totalArea <= 0.0f) {
		return center;
	}

	for (size_t i = 0; i < positions.size(); ++i) {
		center += areas[i]*positions[i];
	}
	return center/totalArea;
}

static EnergyReport evaluateEnergyAndGradient(
	const ExperimentState &state,
	const ExperimentParams &params,
	const std::vector<adsdf::AdsdfPointSample> &samples,
	const std::vector<Landmark> &landmarks,
	std::vector<vec3> *gradPOut,
	std::vector<vec3> *gradQOut) {
	const i32 n = i32(state.p.size());
	const bool needsGradient = gradPOut != nullptr && gradQOut != nullptr;

	if (needsGradient) {
		gradPOut->assign(size_t(n), vec3(0.0f));
		gradQOut->assign(size_t(n), vec3(0.0f));
	}

	EnergyReport report = {};
	f64 weightedOffset2 = 0.0;
	f64 weightedViolation = 0.0;
	f64 weightedLandmark2 = 0.0;
	f64 landmarkWeightSum = 0.0;

	for (i32 i = 0; i < n; ++i) {
		const f32 area = state.vertexAreas[size_t(i)];
		const adsdf::AdsdfPointSample sample = samples[size_t(i)];
		const f32 offsetResidual = sample.phi - params.desiredOffset;
		const f32 violation = sprMax<f32>(0.0f, params.desiredOffset - sample.phi);

		report.minPhi = sprMin<f32>(report.minPhi, sample.phi);
		report.maxViolation = sprMax<f32>(report.maxViolation, violation);
		if (violation > 0.0f) {
			++report.numViolating;
		}
		weightedViolation += f64(area)*f64(violation);
		weightedOffset2 += f64(area)*f64(offsetResidual)*f64(offsetResidual);

		const f64 offsetEnergy = 0.5*double(params.tempOffset)*double(area)*double(offsetResidual)*double(offsetResidual);
		report.offset += offsetEnergy;
		if (needsGradient) {
			(*gradPOut)[size_t(i)] += params.tempOffset*area*offsetResidual*sample.grad;
		}

		const f32 t = params.desiredOffset - sample.phi;
		const f32 rho = stableSoftplus(t, params.softplusEps);
		const f32 rhoPrime = stableSigmoid(t, params.softplusEps);
		const f64 clearEnergy = 0.5*double(params.tempClear)*double(area)*double(rho)*double(rho);
		report.clearance += clearEnergy;
		if (needsGradient) {
			(*gradPOut)[size_t(i)] += -params.tempClear*area*rho*rhoPrime*sample.grad;
		}
	}

	for (Landmark landmark : landmarks) {
		const vec3 residual = state.p[size_t(landmark.vertexIdx)] - landmark.target;
		const f32 weight = params.tempLand*landmark.weight;
		report.landmark += 0.5*double(weight)*double(sqNorm(residual));
		weightedLandmark2 += double(landmark.weight)*double(sqNorm(residual));
		landmarkWeightSum += double(landmark.weight);

		if (needsGradient) {
			(*gradPOut)[size_t(landmark.vertexIdx)] += weight*residual;
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
		const f32 alphaBase =
			params.tempArap*
			state.kappa[size_t(i)]*
			state.vertexAreas[size_t(i)]/
			f32(degree);

		for (i32 k = begin; k < end; ++k) {
			const i32 j = state.topology.neighbors[size_t(k)];
			const vec3 e0 = state.p0[size_t(i)] - state.p0[size_t(j)];
			const vec3 e = state.p[size_t(i)] - state.p[size_t(j)];
			const vec3 r = R*e0 - e;

			report.arap += 0.5*double(alphaBase)*double(sqNorm(r));
			if (needsGradient) {
				(*gradPOut)[size_t(i)] += -alphaBase*r;
				(*gradPOut)[size_t(j)] += alphaBase*r;
				const vec3 localE = Rinv*e;
				(*gradQOut)[size_t(i)] += -2.0f*alphaBase*cross(e0, localE);
			}
		}
	}

	const vec3 center = areaWeightedCenter(state.p, state.vertexAreas, state.totalArea);
	const vec3 centerResidual = center - state.center0;
	report.center = 0.5*double(params.tempCenter)*double(sqNorm(centerResidual));
	if (needsGradient && state.totalArea > 0.0f) {
		for (i32 i = 0; i < n; ++i) {
			const f32 w = state.vertexAreas[size_t(i)]/state.totalArea;
			(*gradPOut)[size_t(i)] += params.tempCenter*w*centerResidual;
		}
	}

	report.total = report.offset + report.clearance + report.landmark + report.arap + report.center;
	if (state.totalArea > 0.0f) {
		report.avgViolation = f32(weightedViolation/double(state.totalArea));
		report.rmsOffset = f32(std::sqrt(weightedOffset2/double(state.totalArea)));
	}
	if (landmarkWeightSum > 0.0) {
		report.rmsLandmark = f32(std::sqrt(weightedLandmark2/landmarkWeightSum));
	}
	return report;
}

static ExperimentState makeInitialState(const geom::Mesh &mesh) {
	ExperimentState state = {};
	state.p0 = mesh.positions;
	state.p = mesh.positions;
	state.q.assign(mesh.positions.size(), quat::one());

	geom::buildMeshTopology(&state.topology, &mesh);
	geom::computeVertexAreas(&state.vertexAreas, &mesh);
	state.totalArea = geom::totalArea(state.vertexAreas);
	state.center0 = areaWeightedCenter(state.p0, state.vertexAreas, state.totalArea);

	state.kappa.assign(mesh.positions.size(), 1.0f);

	return state;
}

static void applyStep(
	ExperimentState *trial,
	const ExperimentState &state,
	const std::vector<vec3> &gradP,
	const std::vector<vec3> &gradQ,
	f32 positionStep,
	f32 rotationStep) {
	*trial = state;
	for (size_t i = 0; i < state.p.size(); ++i) {
		const f32 mass = state.vertexAreas[i] > 1.0e-8f ? state.vertexAreas[i] : 1.0f;
		trial->p[i] = state.p[i] - (positionStep/mass)*gradP[i];
		trial->q[i] = state.q[i]*exp_t((-rotationStep/mass)*gradQ[i]);
		normalize(trial->q[i]);
	}
}

static EnergyReport evaluateWithCurrentSamples(
	adsdf::GpuAdsdfSampler *sampler,
	std::vector<adsdf::AdsdfPointSample> *samples,
	adsdf::AdsdfView field,
	const ExperimentState &state,
	const ExperimentParams &params,
	const std::vector<Landmark> &landmarks,
	std::vector<vec3> *gradP,
	std::vector<vec3> *gradQ) {
	const std::vector<adsdf::AdsdfPointSample> &currentSamples = adsdf::sampleAdsdfPointsBlocking(
		samples,
		sampler,
		field,
		state.p,
		0.003f);
	return evaluateEnergyAndGradient(state, params, currentSamples, landmarks, gradP, gradQ);
}

static void optimize(
	ExperimentState *state,
	const ExperimentParams &params,
	adsdf::GpuAdsdfSampler *sampler,
	std::vector<adsdf::AdsdfPointSample> *samples,
	adsdf::AdsdfView field,
	const std::vector<Landmark> &landmarks,
	expdiag::RunDiagnostics *run) {
	std::vector<vec3> gradP;
	std::vector<vec3> gradQ;

	f32 positionStep = params.positionStep;
	f32 rotationStep = params.rotationStep;
	ArapContinuationState arapContinuation = makeArapContinuation(params);

	for (i32 iter = 0; iter < params.iterations; ++iter) {
		ExperimentParams iterParams = params;
		iterParams.tempArap = arapContinuation.weight;

		const EnergyReport current = evaluateWithCurrentSamples(
			sampler,
			samples,
			field,
			*state,
			iterParams,
			landmarks,
			&gradP,
			&gradQ);

		if (iter % 20 == 0 || iter + 1 == params.iterations) {
			std::printf(
				"iter %3d energy=%10.6f minPhi=% .6f maxViolation=% .6f rmsOffset=% .6f rmsLandmark=% .6f arapW=% .3f\n",
				iter,
				current.total,
				current.minPhi,
				current.maxViolation,
				current.rmsOffset,
				current.rmsLandmark,
				iterParams.tempArap);
		}

		bool accepted = false;
		bool shouldStop = false;
		for (i32 attempt = 0; attempt < 12; ++attempt) {
			ExperimentState trial;
			applyStep(&trial, *state, gradP, gradQ, positionStep, rotationStep);
			const EnergyReport trialReport = evaluateWithCurrentSamples(
				sampler,
				samples,
				field,
				trial,
				iterParams,
				landmarks,
				nullptr,
				nullptr);

			if (trialReport.total <= current.total) {
				*state = std::move(trial);
				const i32 diagnosticIteration = expdiag::nextIteration(*run);
				expdiag::appendSample(run, makeDiagnosticSample(diagnosticIteration, trialReport, iterParams.tempArap));
				maybeRelaxArapContinuation(&arapContinuation, params, diagnosticIteration, trialReport);
				const i32 finalStageAge = diagnosticIteration - arapContinuation.stageStartIteration + 1;
				const bool canStop =
					isFinalArapStage(arapContinuation, params) &&
					finalStageAge >= params.stop.patience + 1;
				shouldStop = canStop && expdiag::recordStopIfReached(run, params.stop);
				positionStep = sprMin<f32>(positionStep*1.03f, params.positionStep);
				rotationStep = sprMin<f32>(rotationStep*1.03f, params.rotationStep);
				accepted = true;
				break;
			}

			positionStep *= 0.5f;
			rotationStep *= 0.5f;
		}

		if (!accepted) {
			if (!isFinalArapStage(arapContinuation, params)) {
				forceRelaxArapContinuation(
					&arapContinuation,
					params,
					expdiag::nextIteration(*run),
					"line-search stall");
				positionStep = params.positionStep;
				rotationStep = params.rotationStep;
				continue;
			}
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
	expdiag::beginRun(&run, "ADSDF body/shirt posed landmark ARAP");

	int deviceCount = 0;
	const cudaError_t deviceCountResult = cudaGetDeviceCount(&deviceCount);
	if (deviceCountResult != cudaSuccess || deviceCount <= 0) {
		cudaGetLastError();
		std::printf("Skipping ADSDF body/shirt landmark experiment: no CUDA device available.\n");
		return 0;
	}

	geom::Mesh body;
	geom::Mesh shirt;
	geom::Mesh posedShirt;
	if (!geom::tryLoadObj(&body, "../main_meshes/body_posed.obj")) {
		std::fprintf(stderr, "failed to load ../main_meshes/body_posed.obj\n");
		return 1;
	}
	if (!geom::tryLoadObj(&shirt, "../main_meshes/shirt.obj")) {
		std::fprintf(stderr, "failed to load ../main_meshes/shirt.obj\n");
		return 1;
	}
	if (!geom::tryLoadObj(&posedShirt, "../main_meshes/shirt_posed.obj")) {
		std::fprintf(stderr, "failed to load ../main_meshes/shirt_posed.obj\n");
		return 1;
	}
	if (!geom::validateMesh(&body) || !geom::validateMesh(&shirt) || !geom::validateMesh(&posedShirt)) {
		std::fprintf(stderr, "body/shirt posed landmark meshes are not valid triangle meshes\n");
		return 1;
	}
	if (shirt.positions.size() != posedShirt.positions.size()) {
		std::fprintf(stderr, "shirt.obj and shirt_posed.obj do not have matching vertex counts\n");
		return 1;
	}
	geom::ensureMeshNormals(&body);
	geom::ensureMeshNormals(&shirt);
	geom::ensureMeshNormals(&posedShirt);

	geom::Bvh bodyBvh;
	geom::buildBvh(&bodyBvh, &body);
	if (bodyBvh.root < 0) {
		std::fprintf(stderr, "failed to build body BVH\n");
		return 1;
	}

	ExperimentParams params = {};
	params.stop.relativeEnergyWindow = 1.0e-5;
	params.stop.absoluteEnergyWindow = 1.0e-7;
	ExperimentState state = makeInitialState(shirt);
	const std::vector<Landmark> landmarks = makeBoundaryLandmarks(shirt, posedShirt);
	if (landmarks.empty()) {
		std::fprintf(stderr, "shirt has no boundary vertices to use as landmarks\n");
		return 1;
	}

	geom::GpuMesh bodyMesh_d = {};
	geom::GpuBvh bodyBvh_d = {};
	geom::initGpuMesh(&bodyMesh_d);
	geom::initGpuBvh(&bodyBvh_d);
	geom::uploadGpuMesh(&bodyMesh_d, &body);
	geom::uploadGpuBvh(&bodyBvh_d, &bodyBvh);

	const geom::Aabb bodyBounds = geom::computeMeshBounds(&body);
	const vec3 bodyExtent = geom::aabbExtent(bodyBounds);
	const f32 maxBodyExtent = sprMax<f32>(bodyExtent.x, sprMax<f32>(bodyExtent.y, bodyExtent.z));
	const vec3 adsdfPadding(0.18f*maxBodyExtent);

	adsdf::AdsdfDesc desc;
	adsdf::initDesc(
		&desc,
		56,
		40,
		64,
		2,
		bodyBounds.lower - adsdfPadding,
		bodyBounds.upper + adsdfPadding,
		adsdf::AdsdfFilterKind::Linear);

	adsdf::AdsdfLinearGrid linearGrid = {};
	adsdf::allocLinearGrid(&linearGrid, &desc);

	adsdf::MeshAdsdfBuildParams buildParams = adsdf::defaultMeshAdsdfBuildParams();
	buildParams.lsq.fineRadius = 2;
	buildParams.lsq.fineExtent = 0.45f;
	buildParams.lsq.regularization = 1.0e-7f;
	buildParams.nearSurfaceEps = 1.0e-4f;
	buildParams.signMethod = geom::MeshSignMethod::RayParity;
	adsdf::buildMeshLsqBlocking(&linearGrid, &bodyMesh_d, &bodyBvh_d, buildParams);

	adsdf::AdsdfTextureGrid textureGrid = {};
	adsdf::allocTextureGrid(&textureGrid, &desc);
	adsdf::uploadTextureGrid(&textureGrid, &linearGrid);
	const adsdf::AdsdfView field = adsdf::makeView(&textureGrid);

	adsdf::GpuAdsdfSampler sampler = {};
	adsdf::initGpuAdsdfSampler(&sampler);
	std::vector<adsdf::AdsdfPointSample> samples;

	auto cleanupCuda = [&]() {
		adsdf::freeGpuAdsdfSampler(&sampler);
		adsdf::freeTextureGrid(&textureGrid);
		adsdf::freeLinearGrid(&linearGrid);
		geom::freeGpuBvh(&bodyBvh_d);
		geom::freeGpuMesh(&bodyMesh_d);
	};

	const EnergyReport originalBefore = evaluateWithCurrentSamples(
		&sampler,
		&samples,
		field,
		state,
		params,
		landmarks,
		nullptr,
		nullptr);
	applyHarmonicRetargetSeed(&state, landmarks, params.seedSmoothingIterations);
	const std::vector<vec3> seededPositions = state.p;
	const EnergyReport before = evaluateWithCurrentSamples(
		&sampler,
		&samples,
		field,
		state,
		params,
		landmarks,
		nullptr,
		nullptr);
	expdiag::appendSample(&run, makeDiagnosticSample(0, before, params.tempArapStart));
	std::printf(
		"original posed body/shirt landmarks: energy=%g minPhi=%g maxViolation=%g avgViolation=%g rmsLandmark=%g violating=%d/%zu\n",
		originalBefore.total,
		originalBefore.minPhi,
		originalBefore.maxViolation,
		originalBefore.avgViolation,
		originalBefore.rmsLandmark,
		originalBefore.numViolating,
		state.p.size());
	std::printf(
		"seeded posed body/shirt landmarks:   energy=%g minPhi=%g maxViolation=%g avgViolation=%g rmsLandmark=%g violating=%d/%zu landmarks=%zu desiredOffset=%g seedIterations=%d\n",
		before.total,
		before.minPhi,
		before.maxViolation,
		before.avgViolation,
		before.rmsLandmark,
		before.numViolating,
		state.p.size(),
		landmarks.size(),
		params.desiredOffset,
		params.seedSmoothingIterations);

	optimize(&state, params, &sampler, &samples, field, landmarks, &run);

	const EnergyReport after = evaluateWithCurrentSamples(
		&sampler,
		&samples,
		field,
		state,
		params,
		landmarks,
		nullptr,
		nullptr);
	std::printf(
		"after posed body/shirt landmarks:  energy=%g minPhi=%g maxViolation=%g avgViolation=%g rmsLandmark=%g violating=%d/%zu\n",
		after.total,
		after.minPhi,
		after.maxViolation,
		after.avgViolation,
		after.rmsLandmark,
		after.numViolating,
		state.p.size());

	if (after.total >= before.total ||
		after.rmsLandmark >= originalBefore.rmsLandmark ||
		after.maxViolation > params.stop.maxViolation ||
		after.numViolating > params.stop.maxViolating) {
		std::fprintf(stderr, "body/shirt landmark experiment did not improve the objective terms\n");
		cleanupCuda();
		return 1;
	}

	geom::Mesh seededShirt = shirt;
	seededShirt.positions = seededPositions;
	geom::ensureMeshNormals(&seededShirt);

	geom::Mesh optimizedShirt = shirt;
	optimizedShirt.positions = state.p;
	geom::ensureMeshNormals(&optimizedShirt);

	const vec3 bodyColor(0.62f, 0.64f, 0.66f);
	const vec3 initialShirtColor(0.95f, 0.30f, 0.18f);
	const vec3 targetShirtColor(0.42f, 0.34f, 0.90f);
	const vec3 optimizedShirtColor(0.88f, 0.14f, 0.22f);

	writeObj("../working/body_shirt_landmark_adsdf_body_posed.obj", body);
	writeObj("../working/body_shirt_landmark_adsdf_shirt_initial.obj", shirt);
	writeObj("../working/body_shirt_landmark_adsdf_shirt_seeded.obj", seededShirt);
	writeObj("../working/body_shirt_landmark_adsdf_shirt_posed_reference.obj", posedShirt);
	writeObj("../working/body_shirt_landmark_adsdf_shirt_optimized.obj", optimizedShirt);
	writeLandmarksObj("../working/body_shirt_landmark_adsdf_targets.obj", landmarks);

	geom::MeshPreviewParams previewParams = geom::defaultMeshPreviewParams(1000, 760);
	const geom::Aabb initialShirtBounds = geom::computeMeshBounds(&shirt);
	const geom::Aabb optimizedShirtBounds = geom::computeMeshBounds(&optimizedShirt);
	const vec3 frontCameraDirection(-2.8f, 0.35f, 1.1f);
	const vec3 shoulderCameraDirection(-0.55f, 2.8f, 0.65f);

	geom::MeshPreviewObject initialPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&body, bodyColor, false),
		geom::makeMeshPreviewObject(&shirt, initialShirtColor, false),
	};
	previewParams.camera = geom::makeFramedPreviewCamera(
		initialShirtBounds,
		frontCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.22f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_initial.ppm", initialPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_initial.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject targetPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&body, bodyColor, false),
		geom::makeMeshPreviewObject(&posedShirt, targetShirtColor, false),
	};
	previewParams.camera = geom::makeFramedPreviewCamera(
		geom::computeMeshBounds(&posedShirt),
		frontCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.22f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_reference.ppm", targetPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_reference.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject seededPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&body, bodyColor, false),
		geom::makeMeshPreviewObject(&seededShirt, targetShirtColor, false),
	};
	previewParams.camera = geom::makeFramedPreviewCamera(
		geom::computeMeshBounds(&seededShirt),
		frontCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.22f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_seeded.ppm", seededPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_seeded.ppm\n");
		cleanupCuda();
		return 1;
	}
	previewParams.camera = geom::makeFramedPreviewCamera(
		geom::computeMeshBounds(&seededShirt),
		shoulderCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.18f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_seeded_shoulder.ppm", seededPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_seeded_shoulder.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject optimizedPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&body, bodyColor, false),
		geom::makeMeshPreviewObject(&optimizedShirt, optimizedShirtColor, false),
	};
	previewParams.camera = geom::makeFramedPreviewCamera(
		optimizedShirtBounds,
		frontCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.22f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_optimized.ppm", optimizedPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_optimized.ppm\n");
		cleanupCuda();
		return 1;
	}
	previewParams.camera = geom::makeFramedPreviewCamera(
		optimizedShirtBounds,
		shoulderCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.18f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_optimized_shoulder.ppm", optimizedPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_optimized_shoulder.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject shirtOnlyObjects[1] = {
		geom::makeMeshPreviewObject(&optimizedShirt, optimizedShirtColor, false),
	};
	previewParams.camera = geom::makeFramedPreviewCamera(
		optimizedShirtBounds,
		frontCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.12f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_shirt_optimized.ppm", shirtOnlyObjects, 1, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_shirt_optimized.ppm\n");
		cleanupCuda();
		return 1;
	}
	previewParams.camera = geom::makeFramedPreviewCamera(
		optimizedShirtBounds,
		shoulderCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.08f);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_shirt_optimized_shoulder.ppm", shirtOnlyObjects, 1, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_shirt_optimized_shoulder.ppm\n");
		cleanupCuda();
		return 1;
	}

	previewParams.camera = geom::makeFramedPreviewCamera(
		optimizedShirtBounds,
		frontCameraDirection,
		previewParams.width,
		previewParams.height,
		vec3(0.0f, 0.0f, 1.0f),
		0.72f,
		1.12f);
	previewParams.shadeMode = geom::MeshPreviewShadeMode::Normal;
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_landmark_adsdf_shirt_optimized_normals.ppm", shirtOnlyObjects, 1, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_shirt_optimized_normals.ppm\n");
		cleanupCuda();
		return 1;
	}

	const vec3 previewCenter = geom::aabbCentroid(bodyBounds);
	const f32 previewRadius = 0.5f*length((bodyBounds.upper + adsdfPadding) - (bodyBounds.lower - adsdfPadding));
	if (!renderAdsdfPreviewPpm(
		"../working/body_shirt_landmark_adsdf_field.ppm",
		field,
		previewCenter,
		previewRadius,
		frontCameraDirection,
		1000,
		760)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf_field.ppm\n");
		cleanupCuda();
		return 1;
	}

	if (!expdiag::writeRunArtifacts("../working/body_shirt_landmark_adsdf", &run)) {
		std::fprintf(stderr, "failed to write body_shirt_landmark_adsdf diagnostics\n");
		cleanupCuda();
		return 1;
	}

	cleanupCuda();

	std::printf("wrote ../working/body_shirt_landmark_adsdf_{body_posed,shirt_initial,shirt_seeded,shirt_posed_reference,shirt_optimized,targets}.obj and preview PPMs\n");
	return 0;
}
