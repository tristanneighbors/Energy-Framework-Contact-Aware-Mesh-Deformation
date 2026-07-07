#pragma once

/**
 * @file adsdf_query.cuh
 * @brief Device-side ADSDF coordinate transforms, coefficient filtering, and queries.
 *
 * Querying interpolates packed coefficient groups from either linear memory or
 * CUDA textures, unpacks them, evaluates the filtered polynomial at global
 * normalized `q`, and returns the polynomial gradient converted to world units.
 * The returned gradient intentionally omits derivatives of interpolation
 * weights; use finite differences when the full reconstructed-field derivative
 * is more important than this cheap normal estimate.
 */

#include "adsdf/adsdf_types.cuh"
#include "adsdf/adsdf_poly.cuh"

namespace adsdf {

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
i32 clampi(i32 x, i32 low, i32 high) {
	return x < low ? low : (x > high ? high : x);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 worldToQ(const AdsdfDesc *desc, vec3 xWorld) {
	return (xWorld - desc->domainCenter)*desc->invDomainHalfExtent;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 qToWorld(const AdsdfDesc *desc, vec3 q) {
	return desc->domainCenter + q*desc->domainHalfExtent;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 worldToGrid(const AdsdfDesc *desc, vec3 xWorld) {
	return (xWorld - desc->domainMin)*desc->invGridSpacing;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gridToWorld(const AdsdfDesc *desc, vec3 g) {
	return desc->domainMin + g*desc->gridSpacing;
}

/** Convert a world-space gradient from polynomial `q` units to world units. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gradQToWorld(const AdsdfDesc *desc, vec3 gradQ) {
	return gradQ*desc->invDomainHalfExtent;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
size_t numNodes(const AdsdfDesc *desc) {
	return size_t(desc->numX)*size_t(desc->numY)*size_t(desc->numZ);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
size_t nodeIdx(const AdsdfDesc *desc, i32 x, i32 y, i32 z) {
	return size_t(x) + size_t(desc->numX)*(size_t(y) + size_t(desc->numY)*size_t(z));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
size_t coeffGroupIdx(const AdsdfDesc *desc, i32 groupIdx, i32 x, i32 y, i32 z) {
	return size_t(groupIdx)*numNodes(desc) + nodeIdx(desc, x, y, z);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 hermiteBlend(f32 t, AdsdfFilterKind filterKind) {
	t = sprClampf(t, 0.0f, 1.0f);

	switch (filterKind) {
		case AdsdfFilterKind::CubicHermite:
			return t*t*(3.0f - 2.0f*t);

		case AdsdfFilterKind::QuinticHermite:
			return t*t*t*(10.0f + t*(-15.0f + 6.0f*t));

		case AdsdfFilterKind::SepticHermite: {
			const f32 t2 = t*t;
			const f32 t4 = t2*t2;
			return t4*(35.0f + t*(-84.0f + t*(70.0f - 20.0f*t)));
		}

		default:
			return t;
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 blendFrac(vec3 f, AdsdfFilterKind filterKind) {
	return vec3(
		hermiteBlend(f.x, filterKind),
		hermiteBlend(f.y, filterKind),
		hermiteBlend(f.z, filterKind));
}

/** Cell origin and clamped unit fraction for interpolation in the regular grid. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void cellBaseAndFrac(
	i32 *baseXOut,
	i32 *baseYOut,
	i32 *baseZOut,
	vec3 *fracOut,
	const AdsdfDesc *desc,
	vec3 xWorld) {
	const vec3 g = worldToGrid(desc, xWorld);

	i32 ix = i32(floorf(g.x));
	i32 iy = i32(floorf(g.y));
	i32 iz = i32(floorf(g.z));

	f32 fx = g.x - f32(ix);
	f32 fy = g.y - f32(iy);
	f32 fz = g.z - f32(iz);

	if (ix < 0) { ix = 0; fx = 0.0f; }
	if (iy < 0) { iy = 0; fy = 0.0f; }
	if (iz < 0) { iz = 0; fz = 0.0f; }

	if (ix >= desc->numX - 1) { ix = desc->numX - 2; fx = 1.0f; }
	if (iy >= desc->numY - 1) { iy = desc->numY - 2; fy = 1.0f; }
	if (iz >= desc->numZ - 1) { iz = desc->numZ - 2; fz = 1.0f; }

	*baseXOut = ix;
	*baseYOut = iy;
	*baseZOut = iz;
	*fracOut = vec3(fx, fy, fz);
}

#if defined(__CUDACC__)

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 vec4FromFloat4(float4 v) {
	return vec4(v.x, v.y, v.z, v.w);
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 fetchNodeGroup(
	const AdsdfView field,
	i32 groupIdx,
	i32 x,
	i32 y,
	i32 z) {
	const AdsdfDesc *desc = &field.desc;
	x = clampi(x, 0, desc->numX - 1);
	y = clampi(y, 0, desc->numY - 1);
	z = clampi(z, 0, desc->numZ - 1);

	if (groupIdx < 0 || groupIdx >= desc->numCoeffGroups) {
		return vec4(0.0f);
	}

	if (field.storageKind == AdsdfStorageKind::Texture3D) {
		const float4 v = tex3D<float4>(
			field.coeffTextures[groupIdx],
			f32(x) + 0.5f,
			f32(y) + 0.5f,
			f32(z) + 0.5f);
		return vec4FromFloat4(v);
	}

	return field.coeffGroups_d[coeffGroupIdx(desc, groupIdx, x, y, z)];
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 fetchGroupNearest(const AdsdfView field, i32 groupIdx, vec3 xWorld) {
	const AdsdfDesc *desc = &field.desc;
	const vec3 g = worldToGrid(desc, xWorld);

	const i32 x = clampi(i32(floorf(g.x + 0.5f)), 0, desc->numX - 1);
	const i32 y = clampi(i32(floorf(g.y + 0.5f)), 0, desc->numY - 1);
	const i32 z = clampi(i32(floorf(g.z + 0.5f)), 0, desc->numZ - 1);

	return fetchNodeGroup(field, groupIdx, x, y, z);
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 trilerpGroupManual(
	const AdsdfView field,
	i32 groupIdx,
	i32 ix,
	i32 iy,
	i32 iz,
	vec3 u) {
	const vec4 c000 = fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0);
	const vec4 c100 = fetchNodeGroup(field, groupIdx, ix + 1, iy + 0, iz + 0);
	const vec4 c010 = fetchNodeGroup(field, groupIdx, ix + 0, iy + 1, iz + 0);
	const vec4 c110 = fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 0);
	const vec4 c001 = fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 1);
	const vec4 c101 = fetchNodeGroup(field, groupIdx, ix + 1, iy + 0, iz + 1);
	const vec4 c011 = fetchNodeGroup(field, groupIdx, ix + 0, iy + 1, iz + 1);
	const vec4 c111 = fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1);

	const vec4 c00 = c000*(1.0f - u.x) + c100*u.x;
	const vec4 c10 = c010*(1.0f - u.x) + c110*u.x;
	const vec4 c01 = c001*(1.0f - u.x) + c101*u.x;
	const vec4 c11 = c011*(1.0f - u.x) + c111*u.x;
	const vec4 c0 = c00*(1.0f - u.y) + c10*u.y;
	const vec4 c1 = c01*(1.0f - u.y) + c11*u.y;
	return c0*(1.0f - u.z) + c1*u.z;
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 fetchGroupLinear(const AdsdfView field, i32 groupIdx, vec3 xWorld) {
	const AdsdfDesc *desc = &field.desc;
	i32 ix;
	i32 iy;
	i32 iz;
	vec3 frac;
	cellBaseAndFrac(&ix, &iy, &iz, &frac, desc, xWorld);

	const vec3 u = blendFrac(frac, desc->filterKind);

	if (field.storageKind == AdsdfStorageKind::Texture3D) {
		const float4 v = tex3D<float4>(
			field.coeffTextures[groupIdx],
			f32(ix) + u.x + 0.5f,
			f32(iy) + u.y + 0.5f,
			f32(iz) + u.z + 0.5f);
		return vec4FromFloat4(v);
	}

	return trilerpGroupManual(field, groupIdx, ix, iy, iz, u);
}

/** Six-tetrahedra cell interpolation used to match the paper's simplex option. */
SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 tetraGroupManual(
	const AdsdfView field,
	i32 groupIdx,
	i32 ix,
	i32 iy,
	i32 iz,
	vec3 u) {
	vec4 out(0.0f);

	if (u.x >= u.y && u.y >= u.z) {
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0)*(1.0f - u.x);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1)*u.z;
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 0, iz + 0)*(u.x - u.y);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 0)*(u.y - u.z);
	} else if (u.x >= u.z && u.z > u.y) {
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0)*(1.0f - u.x);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1)*u.y;
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 0, iz + 0)*(u.x - u.z);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 0, iz + 1)*(u.z - u.y);
	} else if (u.y >= u.x && u.x >= u.z) {
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0)*(1.0f - u.y);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1)*u.z;
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 1, iz + 0)*(u.y - u.x);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 0)*(u.x - u.z);
	} else if (u.y >= u.z && u.z > u.x) {
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0)*(1.0f - u.y);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1)*u.x;
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 1, iz + 0)*(u.y - u.z);
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 1, iz + 1)*(u.z - u.x);
	} else if (u.z >= u.x && u.x >= u.y) {
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0)*(1.0f - u.z);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1)*u.y;
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 1)*(u.z - u.x);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 0, iz + 1)*(u.x - u.y);
	} else {
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 0)*(1.0f - u.z);
		out += fetchNodeGroup(field, groupIdx, ix + 1, iy + 1, iz + 1)*u.x;
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 0, iz + 1)*(u.z - u.y);
		out += fetchNodeGroup(field, groupIdx, ix + 0, iy + 1, iz + 1)*(u.y - u.x);
	}

	return out;
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec4 fetchGroupTetra(const AdsdfView field, i32 groupIdx, vec3 xWorld) {
	const AdsdfDesc *desc = &field.desc;
	i32 ix;
	i32 iy;
	i32 iz;
	vec3 frac;
	cellBaseAndFrac(&ix, &iy, &iz, &frac, desc, xWorld);
	const vec3 u = blendFrac(frac, desc->filterKind);
	return tetraGroupManual(field, groupIdx, ix, iy, iz, u);
}

/** Fetch all coefficient groups under the field's configured filter policy. */
SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
void fetchCoeffGroups(vec4 *groupsOut, const AdsdfView field, vec3 xWorld) {
	const AdsdfDesc *desc = &field.desc;

	for (i32 groupIdx = 0; groupIdx < desc->numCoeffGroups; ++groupIdx) {
		switch (desc->filterKind) {
			case AdsdfFilterKind::Nearest:
				groupsOut[groupIdx] = fetchGroupNearest(field, groupIdx, xWorld);
				break;

			case AdsdfFilterKind::TetraLinear:
				groupsOut[groupIdx] = fetchGroupTetra(field, groupIdx, xWorld);
				break;

			default:
				groupsOut[groupIdx] = fetchGroupLinear(field, groupIdx, xWorld);
				break;
		}
	}

	for (i32 groupIdx = desc->numCoeffGroups; groupIdx < ADSDF_MAX_COEFF_GROUPS; ++groupIdx) {
		groupsOut[groupIdx] = vec4(0.0f);
	}
}

/** Query ADSDF value and cheap polynomial-gradient normal estimate. */
SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
AdsdfQueryResult query(const AdsdfView field, vec3 xWorld) {
	vec4 groups[ADSDF_MAX_COEFF_GROUPS];
	f32 coeffs[ADSDF_MAX_COEFFS];

	fetchCoeffGroups(groups, field, xWorld);
	unpackCoeffGroups(coeffs, groups, field.desc.numCoeffs);

	const vec3 q = worldToQ(&field.desc, xWorld);
	AdsdfQueryResult out;
	out.value = evalPoly(coeffs, field.desc.degree, q);
	out.grad = gradQToWorld(&field.desc, gradPolyQ(coeffs, field.desc.degree, q));
	return out;
}

SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
f32 queryValue(const AdsdfView field, vec3 xWorld) {
	return query(field, xWorld).value;
}

/** Full reconstructed-field normal estimate by central differences. */
SPR_CUDA_DEVICE SPR_CUDA_FORCE_INLINE
vec3 queryNormalFiniteDiff(const AdsdfView field, vec3 xWorld, f32 eps) {
	const vec3 ex(eps, 0.0f, 0.0f);
	const vec3 ey(0.0f, eps, 0.0f);
	const vec3 ez(0.0f, 0.0f, eps);

	vec3 grad;
	grad.x = queryValue(field, xWorld + ex) - queryValue(field, xWorld - ex);
	grad.y = queryValue(field, xWorld + ey) - queryValue(field, xWorld - ey);
	grad.z = queryValue(field, xWorld + ez) - queryValue(field, xWorld - ez);
	return normal(grad);
}

#endif

} // namespace adsdf
