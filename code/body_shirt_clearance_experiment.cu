#include "geometry.cuh"
#include "experiment_diagnostics.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct ExperimentParams {
	f32 desiredOffset = 0.025f;
	f32 softplusEps = 0.008f;

	f32 tempOffset = 0.0f;
	f32 tempClear = 120.0f;
	f32 tempArap = 1.0f;
	f32 tempCenter = 0.02f;

	i32 iterations = 220;
	f32 positionStep = 0.012f;
	f32 rotationStep = 0.02f;
	expdiag::StopCriteria stop;
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
	f64 center = 0.0;

	f32 minPhi = F32_MAX;
	f32 maxViolation = 0.0f;
	f32 avgViolation = 0.0f;
	f32 rmsOffset = 0.0f;
	i32 numViolating = 0;
};

static expdiag::IterationSample makeDiagnosticSample(i32 iteration, const EnergyReport &report) {
	expdiag::IterationSample sample = {};
	sample.iteration = iteration;
	sample.energy = report.total;
	sample.maxViolation = report.maxViolation;
	sample.avgViolation = report.avgViolation;
	sample.minPhi = report.minPhi;
	sample.rmsOffset = report.rmsOffset;
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
	const std::vector<geom::MeshSdfSample> &samples,
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

	for (i32 i = 0; i < n; ++i) {
		const f32 area = state.vertexAreas[size_t(i)];
		const geom::MeshSdfSample sample = samples[size_t(i)];
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

	report.total = report.offset + report.clearance + report.arap + report.center;
	if (state.totalArea > 0.0f) {
		report.avgViolation = f32(weightedViolation/double(state.totalArea));
		report.rmsOffset = f32(std::sqrt(weightedOffset2/double(state.totalArea)));
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
	geom::GpuMeshSdfSampler *sampler,
	std::vector<geom::MeshSdfSample> *samples,
	geom::MeshView target,
	geom::BvhView bvh,
	const ExperimentState &state,
	const ExperimentParams &params,
	std::vector<vec3> *gradP,
	std::vector<vec3> *gradQ) {
	const std::vector<geom::MeshSdfSample> &currentSamples = geom::sampleMeshSdfPointsBlocking(
		samples,
		sampler,
		target,
		bvh,
		state.p,
		geom::MeshSignMethod::RayParity);
	return evaluateEnergyAndGradient(state, params, currentSamples, gradP, gradQ);
}

static void optimize(
	ExperimentState *state,
	const ExperimentParams &params,
	geom::GpuMeshSdfSampler *sampler,
	std::vector<geom::MeshSdfSample> *samples,
	geom::MeshView target,
	geom::BvhView bvh,
	expdiag::RunDiagnostics *run) {
	std::vector<vec3> gradP;
	std::vector<vec3> gradQ;

	f32 positionStep = params.positionStep;
	f32 rotationStep = params.rotationStep;

	for (i32 iter = 0; iter < params.iterations; ++iter) {
		const EnergyReport current = evaluateWithCurrentSamples(
			sampler,
			samples,
			target,
			bvh,
			*state,
			params,
			&gradP,
			&gradQ);

		if (iter % 20 == 0 || iter + 1 == params.iterations) {
			std::printf(
				"iter %3d energy=%10.6f minPhi=% .6f maxViolation=% .6f rmsOffset=% .6f\n",
				iter,
				current.total,
				current.minPhi,
				current.maxViolation,
				current.rmsOffset);
		}

		bool accepted = false;
		bool shouldStop = false;
		for (i32 attempt = 0; attempt < 12; ++attempt) {
			ExperimentState trial;
			applyStep(&trial, *state, gradP, gradQ, positionStep, rotationStep);
			const EnergyReport trialReport = evaluateWithCurrentSamples(
				sampler,
				samples,
				target,
				bvh,
				trial,
				params,
				nullptr,
				nullptr);

			if (trialReport.total <= current.total) {
				*state = std::move(trial);
				expdiag::appendSample(run, makeDiagnosticSample(expdiag::nextIteration(*run), trialReport));
				shouldStop = expdiag::recordStopIfReached(run, params.stop);
				positionStep = sprMin<f32>(positionStep*1.03f, params.positionStep);
				rotationStep = sprMin<f32>(rotationStep*1.03f, params.rotationStep);
				accepted = true;
				break;
			}

			positionStep *= 0.5f;
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
	expdiag::beginRun(&run, "Body/shirt ARAP clearance");

	geom::Mesh body;
	geom::Mesh shirt;
	if (!geom::tryLoadObj(&body, "../main_meshes/body.obj")) {
		std::fprintf(stderr, "failed to load ../main_meshes/body.obj\n");
		return 1;
	}
	if (!geom::tryLoadObj(&shirt, "../main_meshes/shirt.obj")) {
		std::fprintf(stderr, "failed to load ../main_meshes/shirt.obj\n");
		return 1;
	}
	if (!geom::validateMesh(&body) || !geom::validateMesh(&shirt)) {
		std::fprintf(stderr, "body/shirt meshes are not valid triangle meshes\n");
		return 1;
	}
	geom::ensureMeshNormals(&body);
	geom::ensureMeshNormals(&shirt);

	geom::Bvh bodyBvh;
	geom::buildBvh(&bodyBvh, &body);
	if (bodyBvh.root < 0) {
		std::fprintf(stderr, "failed to build body BVH\n");
		return 1;
	}

	ExperimentParams params = {};
	ExperimentState state = makeInitialState(shirt);

	geom::GpuMesh bodyMesh_d = {};
	geom::GpuBvh bodyBvh_d = {};
	geom::initGpuMesh(&bodyMesh_d);
	geom::initGpuBvh(&bodyBvh_d);
	geom::uploadGpuMesh(&bodyMesh_d, &body);
	geom::uploadGpuBvh(&bodyBvh_d, &bodyBvh);

	geom::GpuMeshSdfSampler sampler = {};
	geom::initGpuMeshSdfSampler(&sampler);
	std::vector<geom::MeshSdfSample> samples;

	auto cleanupCuda = [&]() {
		geom::freeGpuMeshSdfSampler(&sampler);
		geom::freeGpuBvh(&bodyBvh_d);
		geom::freeGpuMesh(&bodyMesh_d);
	};

	const geom::MeshView bodyView = geom::viewGpuMesh(&bodyMesh_d);
	const geom::BvhView bvhView = geom::viewGpuBvh(&bodyBvh_d);

	const EnergyReport before = evaluateWithCurrentSamples(&sampler, &samples, bodyView, bvhView, state, params, nullptr, nullptr);
	expdiag::appendSample(&run, makeDiagnosticSample(0, before));
	std::printf(
		"before body/shirt: energy=%g minPhi=%g maxViolation=%g avgViolation=%g violating=%d/%zu desiredOffset=%g\n",
		before.total,
		before.minPhi,
		before.maxViolation,
		before.avgViolation,
		before.numViolating,
		state.p.size(),
		params.desiredOffset);

	optimize(&state, params, &sampler, &samples, bodyView, bvhView, &run);

	const EnergyReport after = evaluateWithCurrentSamples(&sampler, &samples, bodyView, bvhView, state, params, nullptr, nullptr);
	std::printf(
		"after body/shirt:  energy=%g minPhi=%g maxViolation=%g avgViolation=%g violating=%d/%zu\n",
		after.total,
		after.minPhi,
		after.maxViolation,
		after.avgViolation,
		after.numViolating,
		state.p.size());

	if (after.total >= before.total || after.maxViolation > before.maxViolation || after.numViolating > before.numViolating) {
		std::fprintf(stderr, "body/shirt clearance experiment did not improve contact\n");
		cleanupCuda();
		return 1;
	}

	geom::Mesh optimizedShirt = shirt;
	optimizedShirt.positions = state.p;
	geom::ensureMeshNormals(&optimizedShirt);

	const vec3 bodyColor(0.62f, 0.64f, 0.66f);
	const vec3 initialShirtColor(0.95f, 0.30f, 0.18f);
	const vec3 optimizedShirtColor(0.88f, 0.14f, 0.22f);

	writeObj("../working/body_shirt_clearance_body.obj", body);
	writeObj("../working/body_shirt_clearance_shirt_initial.obj", shirt);
	writeObj("../working/body_shirt_clearance_shirt_optimized.obj", optimizedShirt);

	geom::MeshPreviewParams previewParams = geom::defaultMeshPreviewParams(1000, 760);
	const geom::Aabb initialShirtBounds = geom::computeMeshBounds(&shirt);
	const geom::Aabb optimizedShirtBounds = geom::computeMeshBounds(&optimizedShirt);
	const vec3 frontCameraDirection(-2.8f, 0.35f, 1.1f);

	geom::MeshPreviewObject initialPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&body, bodyColor, false),
		geom::makeMeshPreviewObject(&shirt, initialShirtColor, false),
	};
	previewParams.camera = geom::makeOrbitPreviewCamera(
		initialShirtBounds,
		frontCameraDirection);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_clearance_initial.ppm", initialPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_clearance_initial.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject optimizedPreviewObjects[2] = {
		geom::makeMeshPreviewObject(&body, bodyColor, false),
		geom::makeMeshPreviewObject(&optimizedShirt, optimizedShirtColor, false),
	};
	previewParams.camera = geom::makeOrbitPreviewCamera(
		optimizedShirtBounds,
		frontCameraDirection);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_clearance_optimized.ppm", optimizedPreviewObjects, 2, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_clearance_optimized.ppm\n");
		cleanupCuda();
		return 1;
	}

	geom::MeshPreviewObject shirtOnlyObjects[1] = {
		geom::makeMeshPreviewObject(&optimizedShirt, optimizedShirtColor, false),
	};
	previewParams.camera = geom::makeOrbitPreviewCamera(
		optimizedShirtBounds,
		frontCameraDirection);
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_clearance_shirt_optimized.ppm", shirtOnlyObjects, 1, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_clearance_shirt_optimized.ppm\n");
		cleanupCuda();
		return 1;
	}

	previewParams.shadeMode = geom::MeshPreviewShadeMode::Normal;
	if (!geom::renderMeshPreviewPpm("../working/body_shirt_clearance_shirt_optimized_normals.ppm", shirtOnlyObjects, 1, previewParams)) {
		std::fprintf(stderr, "failed to write body_shirt_clearance_shirt_optimized_normals.ppm\n");
		cleanupCuda();
		return 1;
	}

	if (!expdiag::writeRunArtifacts("../working/body_shirt_clearance", &run)) {
		std::fprintf(stderr, "failed to write body_shirt_clearance diagnostics\n");
		cleanupCuda();
		return 1;
	}

	cleanupCuda();

	std::printf("wrote ../working/body_shirt_clearance_{body,shirt_initial,shirt_optimized}.obj and preview PPMs\n");
	return 0;
}
