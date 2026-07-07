#pragma once

/**
 * @file adsdf_poly.cuh
 * @brief Degree-limited normalized power-basis polynomial helpers.
 *
 * Coefficients are ordered by total degree through degree 3 and packed four at
 * a time for texture storage. The basis variable is global normalized `q`, not
 * a node-local offset, because filtered coefficient groups must remain valid
 * before polynomial evaluation.
 */

#include "adsdf/adsdf_types.cuh"

namespace adsdf {

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
i32 numCoeffs(i32 degree) {
	switch (degree) {
		case 0: return 1;
		case 1: return 4;
		case 2: return 10;
		case 3: return 20;
		default: return 0;
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
i32 numCoeffGroups(i32 degree) {
	return (numCoeffs(degree) + 3)/4;
}

/** Exponent triple for the fixed coefficient ordering. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
AdsdfMultiIndex3 coeffExp(i32 coeffIdx) {
	switch (coeffIdx) {
		case 0: return {0, 0, 0};

		case 1: return {1, 0, 0};
		case 2: return {0, 1, 0};
		case 3: return {0, 0, 1};

		case 4: return {2, 0, 0};
		case 5: return {1, 1, 0};
		case 6: return {1, 0, 1};
		case 7: return {0, 2, 0};
		case 8: return {0, 1, 1};
		case 9: return {0, 0, 2};

		case 10: return {3, 0, 0};
		case 11: return {2, 1, 0};
		case 12: return {2, 0, 1};
		case 13: return {1, 2, 0};
		case 14: return {1, 1, 1};
		case 15: return {1, 0, 2};
		case 16: return {0, 3, 0};
		case 17: return {0, 2, 1};
		case 18: return {0, 1, 2};
		case 19: return {0, 0, 3};

		default: return {-1, -1, -1};
	}
}

/** Fixed-order coefficient lookup for exponents with total degree <= 3. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
i32 coeffIndexFromExp(i32 x, i32 y, i32 z) {
	const i32 degree = x + y + z;

	if (x < 0 || y < 0 || z < 0 || degree > ADSDF_MAX_DEGREE) {
		return -1;
	}

	if (degree == 0) {
		return 0;
	}

	if (degree == 1) {
		if (x == 1) { return 1; }
		if (y == 1) { return 2; }
		if (z == 1) { return 3; }
	}

	if (degree == 2) {
		if (x == 2) { return 4; }
		if (x == 1 && y == 1) { return 5; }
		if (x == 1 && z == 1) { return 6; }
		if (y == 2) { return 7; }
		if (y == 1 && z == 1) { return 8; }
		if (z == 2) { return 9; }
	}

	if (degree == 3) {
		if (x == 3) { return 10; }
		if (x == 2 && y == 1) { return 11; }
		if (x == 2 && z == 1) { return 12; }
		if (x == 1 && y == 2) { return 13; }
		if (x == 1 && y == 1 && z == 1) { return 14; }
		if (x == 1 && z == 2) { return 15; }
		if (y == 3) { return 16; }
		if (y == 2 && z == 1) { return 17; }
		if (y == 1 && z == 2) { return 18; }
		if (z == 3) { return 19; }
	}

	return -1;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 powSmall(f32 x, i32 n) {
	switch (n) {
		case 0: return 1.0f;
		case 1: return x;
		case 2: return x*x;
		case 3: return x*x*x;
		default: return 0.0f;
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 binomSmall(i32 n, i32 k) {
	if (k < 0 || k > n) {
		return 0.0f;
	}

	if (k == 0 || k == n) {
		return 1.0f;
	}

	if (n == 2) {
		return 2.0f;
	}

	if (n == 3) {
		return k == 1 || k == 2 ? 3.0f : 1.0f;
	}

	return 1.0f;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void zeroCoeffs(f32 *coeffsOut) {
	SPR_UNROLL
	for (i32 i = 0; i < ADSDF_MAX_COEFFS; ++i) {
		coeffsOut[i] = 0.0f;
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 basisValue(i32 coeffIdx, vec3 q) {
	const AdsdfMultiIndex3 e = coeffExp(coeffIdx);
	return powSmall(q.x, e.x)*powSmall(q.y, e.y)*powSmall(q.z, e.z);
}

/** Pack up to four scalar coefficients into the group layout used by storage. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec4 coeffGroupFromArray(const f32 *coeffs, i32 groupIdx, i32 numCoeffsIn) {
	vec4 out(0.0f);
	const i32 base = 4*groupIdx;

	if (base + 0 < numCoeffsIn) { out.x = coeffs[base + 0]; }
	if (base + 1 < numCoeffsIn) { out.y = coeffs[base + 1]; }
	if (base + 2 < numCoeffsIn) { out.z = coeffs[base + 2]; }
	if (base + 3 < numCoeffsIn) { out.w = coeffs[base + 3]; }

	return out;
}

/** Expand packed coefficient groups into a full scratch array. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void unpackCoeffGroups(f32 *coeffsOut, const vec4 *groups, i32 numCoeffsIn) {
	zeroCoeffs(coeffsOut);

	const i32 groupCount = (numCoeffsIn + 3)/4;
	for (i32 groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
		const vec4 g = groups[groupIdx];
		const i32 base = 4*groupIdx;

		if (base + 0 < numCoeffsIn) { coeffsOut[base + 0] = g.x; }
		if (base + 1 < numCoeffsIn) { coeffsOut[base + 1] = g.y; }
		if (base + 2 < numCoeffsIn) { coeffsOut[base + 2] = g.z; }
		if (base + 3 < numCoeffsIn) { coeffsOut[base + 3] = g.w; }
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalPolyD0(const f32 *c) {
	return c[0];
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalPolyD1(const f32 *c, vec3 q) {
	return fmaf(c[1], q.x, fmaf(c[2], q.y, fmaf(c[3], q.z, c[0])));
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalPolyD2(const f32 *c, vec3 q) {
	const f32 x = q.x;
	const f32 y = q.y;
	const f32 z = q.z;

	f32 out = evalPolyD1(c, q);
	out = fmaf(c[4], x*x, out);
	out = fmaf(c[5], x*y, out);
	out = fmaf(c[6], x*z, out);
	out = fmaf(c[7], y*y, out);
	out = fmaf(c[8], y*z, out);
	out = fmaf(c[9], z*z, out);
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalPolyD3(const f32 *c, vec3 q) {
	const f32 x = q.x;
	const f32 y = q.y;
	const f32 z = q.z;
	const f32 x2 = x*x;
	const f32 y2 = y*y;
	const f32 z2 = z*z;

	f32 out = evalPolyD2(c, q);
	out = fmaf(c[10], x2*x, out);
	out = fmaf(c[11], x2*y, out);
	out = fmaf(c[12], x2*z, out);
	out = fmaf(c[13], x*y2, out);
	out = fmaf(c[14], x*y*z, out);
	out = fmaf(c[15], x*z2, out);
	out = fmaf(c[16], y2*y, out);
	out = fmaf(c[17], y2*z, out);
	out = fmaf(c[18], y*z2, out);
	out = fmaf(c[19], z2*z, out);
	return out;
}

/** Evaluate a degree-limited polynomial at global normalized coordinate `q`. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
f32 evalPoly(const f32 *c, i32 degree, vec3 q) {
	switch (degree) {
		case 0: return evalPolyD0(c);
		case 1: return evalPolyD1(c, q);
		case 2: return evalPolyD2(c, q);
		case 3: return evalPolyD3(c, q);
		default: return 0.0f;
	}
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gradPolyD0() {
	return vec3(0.0f);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gradPolyD1(const f32 *c) {
	return vec3(c[1], c[2], c[3]);
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gradPolyD2(const f32 *c, vec3 q) {
	const f32 x = q.x;
	const f32 y = q.y;
	const f32 z = q.z;

	vec3 out;
	out.x = c[1] + 2.0f*c[4]*x + c[5]*y + c[6]*z;
	out.y = c[2] + c[5]*x + 2.0f*c[7]*y + c[8]*z;
	out.z = c[3] + c[6]*x + c[8]*y + 2.0f*c[9]*z;
	return out;
}

SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gradPolyD3(const f32 *c, vec3 q) {
	const f32 x = q.x;
	const f32 y = q.y;
	const f32 z = q.z;
	const f32 x2 = x*x;
	const f32 y2 = y*y;
	const f32 z2 = z*z;

	vec3 out = gradPolyD2(c, q);

	out.x += 3.0f*c[10]*x2 + 2.0f*c[11]*x*y + 2.0f*c[12]*x*z;
	out.x += c[13]*y2 + c[14]*y*z + c[15]*z2;

	out.y += c[11]*x2 + 2.0f*c[13]*x*y + c[14]*x*z;
	out.y += 3.0f*c[16]*y2 + 2.0f*c[17]*y*z + c[18]*z2;

	out.z += c[12]*x2 + c[14]*x*y + 2.0f*c[15]*x*z;
	out.z += c[17]*y2 + 2.0f*c[18]*y*z + 3.0f*c[19]*z2;

	return out;
}

/** Gradient with respect to global normalized coordinate `q`. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
vec3 gradPolyQ(const f32 *c, i32 degree, vec3 q) {
	switch (degree) {
		case 0: return gradPolyD0();
		case 1: return gradPolyD1(c);
		case 2: return gradPolyD2(c, q);
		case 3: return gradPolyD3(c, q);
		default: return vec3(0.0f);
	}
}

/** Translate coefficients from powers of `(q - centerQ)` into powers of `q`. */
SPR_CUDA_HOST_DEVICE_FORCE_INLINE
void translateLocalToGlobal(
	f32 *globalOut,
	const f32 *local,
	i32 degree,
	vec3 centerQ) {
	zeroCoeffs(globalOut);

	const i32 count = numCoeffs(degree);
	for (i32 alphaIdx = 0; alphaIdx < count; ++alphaIdx) {
		const AdsdfMultiIndex3 a = coeffExp(alphaIdx);
		const f32 localCoeff = local[alphaIdx];

		for (i32 bx = 0; bx <= a.x; ++bx) {
			for (i32 by = 0; by <= a.y; ++by) {
				for (i32 bz = 0; bz <= a.z; ++bz) {
					const i32 betaIdx = coeffIndexFromExp(bx, by, bz);
					if (betaIdx < 0 || betaIdx >= count) {
						continue;
					}

					const f32 factor =
						binomSmall(a.x, bx)*powSmall(-centerQ.x, a.x - bx)*
						binomSmall(a.y, by)*powSmall(-centerQ.y, a.y - by)*
						binomSmall(a.z, bz)*powSmall(-centerQ.z, a.z - bz);

					globalOut[betaIdx] += localCoeff*factor;
				}
			}
		}
	}
}

} // namespace adsdf
