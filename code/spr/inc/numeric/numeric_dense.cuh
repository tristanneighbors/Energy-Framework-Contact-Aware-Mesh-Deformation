#pragma once

/**
 * @file numeric_dense.cuh
 * @brief Small host-side dense linear algebra helpers for prototype solvers.
 */

#include "spr_global_include.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace numeric {

struct DenseDampingParams {
	f64 lambda = 1.0e-3;
	f64 minDiag = 1.0e-8;
	f64 pivotEps = 1.0e-14;
};

struct DenseLmStepParams {
	f64 lambda = 1.0e-3;
	f64 minLambda = 1.0e-8;
	f64 maxLambda = 1.0e12;
	f64 lambdaDecrease = 0.35;
	f64 lambdaIncrease = 8.0;
	f64 maxBlockNorm = 0.0;
	i32 blockSize = 0;
	DenseDampingParams damping = {};
};

struct DenseLmStepResult {
	bool solved = false;
	bool clamped = false;
	f64 lambda = 0.0;
	f64 maxBlockNorm = 0.0;
};

inline void zeroDenseNormalSystem(std::vector<f64> *jtJ, std::vector<f64> *jtR, i32 n) {
	if (jtJ != nullptr) {
		jtJ->assign(size_t(n)*size_t(n), 0.0);
	}
	if (jtR != nullptr) {
		jtR->assign(size_t(n), 0.0);
	}
}

inline void accumulateDenseResidual(
	std::vector<f64> *jtJ,
	std::vector<f64> *jtR,
	f64 *residual2,
	const std::vector<f64> &j,
	f64 r) {
	SPR_ASSERT(jtJ != nullptr);
	SPR_ASSERT(jtR != nullptr);
	SPR_ASSERT(residual2 != nullptr);

	const i32 n = i32(j.size());
	SPR_ASSERT(i32(jtJ->size()) == n*n);
	SPR_ASSERT(i32(jtR->size()) == n);

	*residual2 += r*r;

	for (i32 row = 0; row < n; ++row) {
		if (j[size_t(row)] == 0.0) {
			continue;
		}

		(*jtR)[size_t(row)] += j[size_t(row)]*r;
		for (i32 col = 0; col < n; ++col) {
			(*jtJ)[size_t(row*n + col)] += j[size_t(row)]*j[size_t(col)];
		}
	}
}

inline bool solveDenseLinearSystem(
	std::vector<f64> a,
	std::vector<f64> b,
	std::vector<f64> *xOut,
	i32 n,
	f64 pivotEps = 1.0e-14) {
	if (xOut == nullptr || n <= 0 || i32(a.size()) != n*n || i32(b.size()) != n) {
		return false;
	}

	for (i32 col = 0; col < n; ++col) {
		i32 pivot = col;
		f64 pivotAbs = std::fabs(a[size_t(col*n + col)]);
		for (i32 row = col + 1; row < n; ++row) {
			const f64 v = std::fabs(a[size_t(row*n + col)]);
			if (v > pivotAbs) {
				pivot = row;
				pivotAbs = v;
			}
		}

		if (pivotAbs < pivotEps) {
			return false;
		}

		if (pivot != col) {
			for (i32 k = col; k < n; ++k) {
				std::swap(a[size_t(col*n + k)], a[size_t(pivot*n + k)]);
			}
			std::swap(b[size_t(col)], b[size_t(pivot)]);
		}

		for (i32 row = col + 1; row < n; ++row) {
			const f64 factor = a[size_t(row*n + col)]/a[size_t(col*n + col)];
			if (factor == 0.0) {
				continue;
			}

			a[size_t(row*n + col)] = 0.0;
			for (i32 k = col + 1; k < n; ++k) {
				a[size_t(row*n + k)] -= factor*a[size_t(col*n + k)];
			}
			b[size_t(row)] -= factor*b[size_t(col)];
		}
	}

	xOut->assign(size_t(n), 0.0);
	for (i32 row = n - 1; row >= 0; --row) {
		f64 rhs = b[size_t(row)];
		for (i32 col = row + 1; col < n; ++col) {
			rhs -= a[size_t(row*n + col)]*(*xOut)[size_t(col)];
		}

		const f64 diag = a[size_t(row*n + row)];
		if (std::fabs(diag) < pivotEps) {
			return false;
		}
		(*xOut)[size_t(row)] = rhs/diag;
	}

	return true;
}

inline bool solveDenseDampedNormalStep(
	std::vector<f64> *deltaOut,
	const std::vector<f64> &jtJ,
	const std::vector<f64> &jtR,
	i32 n,
	DenseDampingParams params) {
	if (deltaOut == nullptr || n <= 0 || i32(jtJ.size()) != n*n || i32(jtR.size()) != n) {
		return false;
	}

	std::vector<f64> a = jtJ;
	std::vector<f64> rhs(size_t(n), 0.0);
	for (i32 row = 0; row < n; ++row) {
		const f64 diag = std::max(std::fabs(jtJ[size_t(row*n + row)]), params.minDiag);
		a[size_t(row*n + row)] += params.lambda*diag;
		rhs[size_t(row)] = -jtR[size_t(row)];
	}

	return solveDenseLinearSystem(a, rhs, deltaOut, n, params.pivotEps);
}

inline f64 maxDenseBlockNorm2(const std::vector<f64> &x, i32 blockSize) {
	if (blockSize <= 0 || x.empty()) {
		return 0.0;
	}

	f64 result = 0.0;
	const i32 numBlocks = i32(x.size())/blockSize;
	for (i32 block = 0; block < numBlocks; ++block) {
		f64 norm2 = 0.0;
		for (i32 i = 0; i < blockSize; ++i) {
			const f64 v = x[size_t(block*blockSize + i)];
			norm2 += v*v;
		}
		result = std::max(result, norm2);
	}
	return result;
}

inline void clampDenseStepByBlockNorm(std::vector<f64> *x, i32 blockSize, f64 maxNorm) {
	if (x == nullptr || blockSize <= 0 || maxNorm <= 0.0) {
		return;
	}

	const f64 maxNorm2 = maxDenseBlockNorm2(*x, blockSize);
	if (maxNorm2 <= maxNorm*maxNorm) {
		return;
	}

	const f64 scale = maxNorm/std::sqrt(maxNorm2);
	for (f64 &v : *x) {
		v *= scale;
	}
}

inline bool solveDenseLmStep(
	std::vector<f64> *deltaOut,
	DenseLmStepResult *resultOut,
	const std::vector<f64> &jtJ,
	const std::vector<f64> &jtR,
	i32 n,
	DenseLmStepParams params) {
	if (resultOut != nullptr) {
		*resultOut = {};
		resultOut->lambda = params.lambda;
	}
	if (deltaOut == nullptr) {
		return false;
	}

	params.lambda = std::max(params.minLambda, std::min(params.lambda, params.maxLambda));
	params.damping.lambda = params.lambda;

	std::vector<f64> delta;
	if (!solveDenseDampedNormalStep(&delta, jtJ, jtR, n, params.damping)) {
		return false;
	}

	const f64 beforeClampNorm2 =
		(params.blockSize > 0) ? maxDenseBlockNorm2(delta, params.blockSize) : 0.0;
	const f64 beforeClampNorm = beforeClampNorm2 > 0.0 ? std::sqrt(beforeClampNorm2) : 0.0;
	bool clamped = false;
	if (params.blockSize > 0 && params.maxBlockNorm > 0.0 && beforeClampNorm > params.maxBlockNorm) {
		clampDenseStepByBlockNorm(&delta, params.blockSize, params.maxBlockNorm);
		clamped = true;
	}

	if (resultOut != nullptr) {
		resultOut->solved = true;
		resultOut->clamped = clamped;
		resultOut->lambda = params.lambda;
		resultOut->maxBlockNorm = clamped ? params.maxBlockNorm : beforeClampNorm;
	}
	*deltaOut = std::move(delta);
	return true;
}

inline void decreaseDenseLmLambda(f64 *lambda, const DenseLmStepParams &params) {
	if (lambda == nullptr) {
		return;
	}

	*lambda = std::max(params.minLambda, (*lambda)*params.lambdaDecrease);
}

inline void increaseDenseLmLambda(f64 *lambda, const DenseLmStepParams &params) {
	if (lambda == nullptr) {
		return;
	}

	*lambda = std::min(params.maxLambda, (*lambda)*params.lambdaIncrease);
}

} // namespace numeric
