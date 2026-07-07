#pragma once

/**
 * @file adsdf_types.cuh
 * @brief Core ADSDF data structures and shared constants.
 *
 * The public data model is deliberately POD-like. `AdsdfDesc` defines the
 * lattice, normalized domain, polynomial degree, and filter policy. Runtime
 * storage is then exposed either as raw linear coefficient groups or CUDA
 * texture objects, with `AdsdfView` providing the device-query view over either
 * representation.
 */

#include "spr_global_include.h"
#include "spr_math.cuh"

#include <cuda_runtime.h>

namespace adsdf {

inline constexpr i32 ADSDF_MAX_DEGREE = 3;
inline constexpr i32 ADSDF_MAX_COEFFS = 20;
inline constexpr i32 ADSDF_MAX_COEFF_GROUPS = 5;
inline constexpr i32 ADSDF_DEFAULT_THREADS_PER_BLOCK = 256;

static_assert(sizeof(vec4) == sizeof(float4), "ADSDF texture upload assumes vec4 and float4 have equal size.");
static_assert(alignof(vec4) >= alignof(float4), "ADSDF texture upload assumes vec4 alignment is compatible with float4.");

enum class AdsdfStorageKind : i32 {
	Linear,
	Texture3D,
};

enum class AdsdfFilterKind : i32 {
	Nearest,
	Linear,
	CubicHermite,
	QuinticHermite,
	SepticHermite,
	TetraLinear,
};

enum class ProceduralSdfKind : i32 {
	Sphere,
	Box,
	Torus,
};

/** Regular lattice, polynomial, coordinate-transform, and filter metadata. */
struct AdsdfDesc {
	i32 numX;
	i32 numY;
	i32 numZ;

	i32 degree;
	i32 numCoeffs;
	i32 numCoeffGroups;

	vec3 domainMin;
	vec3 domainMax;
	vec3 domainCenter;
	vec3 domainHalfExtent;
	vec3 invDomainHalfExtent;

	vec3 gridSpacing;
	vec3 invGridSpacing;
	vec3 gridSpacingQ;

	AdsdfFilterKind filterKind;
};

/** Construction/debug storage: group-major device array of packed coefficients. */
struct AdsdfLinearGrid {
	AdsdfDesc desc;

	// Group-major storage:
	// coeffGroups_d[groupIdx*numNodes + nodeIdx].
	// Each vec4 stores four polynomial coefficients.
	vec4 *coeffGroups_d;
};

/** Texture query storage: one CUDA 3D array/texture object per coefficient group. */
struct AdsdfTextureGrid {
	AdsdfDesc desc;

	cudaArray_t coeffArrays[ADSDF_MAX_COEFF_GROUPS];
	cudaTextureObject_t coeffTextures[ADSDF_MAX_COEFF_GROUPS];
};

/** Device-side field view used by query and render kernels. */
struct AdsdfView {
	AdsdfDesc desc;
	AdsdfStorageKind storageKind;

	const vec4 *coeffGroups_d;
	cudaTextureObject_t coeffTextures[ADSDF_MAX_COEFF_GROUPS];
};

/** Query value plus gradient of the filtered polynomial in world coordinates. */
struct AdsdfQueryResult {
	f32 value;
	vec3 grad;
};

/** Parameters for the local least-squares fit used during procedural construction. */
struct AdsdfLsqParams {
	i32 fineRadius;
	f32 fineExtent;
	f32 regularization;
};

/** Packed procedural SDF parameters for simple smoke-test and construction sources. */
struct ProceduralSdf {
	ProceduralSdfKind kind;

	// Sphere: a.xyz = center, a.w = radius.
	// Box:    a.xyz = center, b.xyz = half extent.
	// Torus:  a.xyz = center, a.w = major radius, b.x = minor radius.
	vec4 a;
	vec4 b;
	vec4 c;
};

struct AdsdfCamera {
	vec3 pos;
	vec3 dir;
	vec3 right;
	vec3 up;

	f32 tanHalfFovY;
	f32 aspect;
};

struct AdsdfRayMarchParams {
	i32 imageWidth;
	i32 imageHeight;

	i32 maxSteps;
	f32 hitEps;
	f32 normalEps;
	f32 tMax;
	f32 stepScale;
	f32 minStep;
	f32 maxStep;

	vec3 lightDir;
};

struct AdsdfRenderTarget {
	u32 *pixels_d;
	i32 width;
	i32 height;
};

struct AdsdfMultiIndex3 {
	i32 x;
	i32 y;
	i32 z;
};

} // namespace adsdf
