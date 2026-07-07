#pragma once

/**
 * @file numeric_sparse.cuh
 * @brief Small host-side sparse normal-equation helpers for prototype solvers.
 */

#include "numeric_dense.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace numeric {

struct SparseTriplet {
	i32 row = 0;
	i32 col = 0;
	f64 value = 0.0;
};

struct SparseEntry {
	i32 index = 0;
	f64 value = 0.0;
};

struct CsrMatrix {
	i32 numRows = 0;
	i32 numCols = 0;
	std::vector<i32> rowOffsets;
	std::vector<i32> colIndices;
	std::vector<f64> values;
};

struct PcgParams {
	i32 maxIterations = 512;
	f64 relativeTolerance = 1.0e-8;
	f64 absoluteTolerance = 1.0e-12;
	f64 denominatorEps = 1.0e-30;
};

struct PcgResult {
	bool converged = false;
	i32 iterations = 0;
	f64 initialResidualNorm = 0.0;
	f64 finalResidualNorm = 0.0;
};

struct SparseLmStepParams {
	f64 lambda = 1.0e-3;
	f64 minLambda = 1.0e-8;
	f64 maxLambda = 1.0e12;
	f64 lambdaDecrease = 0.35;
	f64 lambdaIncrease = 8.0;
	f64 minDiag = 1.0e-8;
	f64 maxBlockNorm = 0.0;
	i32 blockSize = 0;
	PcgParams pcg = {};
};

struct SparseLmStepResult {
	bool solved = false;
	bool clamped = false;
	bool converged = false;
	i32 iterations = 0;
	f64 lambda = 0.0;
	f64 maxBlockNorm = 0.0;
	f64 initialResidualNorm = 0.0;
	f64 finalResidualNorm = 0.0;
};

inline void zeroSparseNormalSystem(std::vector<SparseTriplet> *jtJ, std::vector<f64> *jtR, i32 n) {
	if (jtJ != nullptr) {
		jtJ->clear();
	}
	if (jtR != nullptr) {
		jtR->assign(size_t(n), 0.0);
	}
}

inline void accumulateSparseResidual(
	std::vector<SparseTriplet> *jtJ,
	std::vector<f64> *jtR,
	f64 *residual2,
	const std::vector<SparseEntry> &j,
	f64 r) {
	SPR_ASSERT(jtJ != nullptr);
	SPR_ASSERT(jtR != nullptr);
	SPR_ASSERT(residual2 != nullptr);

	*residual2 += r*r;
	for (SparseEntry rowEntry : j) {
		if (rowEntry.value == 0.0) {
			continue;
		}

		SPR_ASSERT(rowEntry.index >= 0 && rowEntry.index < i32(jtR->size()));
		(*jtR)[size_t(rowEntry.index)] += rowEntry.value*r;
		for (SparseEntry colEntry : j) {
			if (colEntry.value == 0.0) {
				continue;
			}
			SPR_ASSERT(colEntry.index >= 0 && colEntry.index < i32(jtR->size()));
			SparseTriplet triplet = {};
			triplet.row = rowEntry.index;
			triplet.col = colEntry.index;
			triplet.value = rowEntry.value*colEntry.value;
			jtJ->push_back(triplet);
		}
	}
}

inline void compressTripletsToCsr(CsrMatrix *matrixOut, std::vector<SparseTriplet> triplets, i32 numRows, i32 numCols) {
	if (matrixOut == nullptr) {
		return;
	}

	matrixOut->numRows = numRows;
	matrixOut->numCols = numCols;
	matrixOut->rowOffsets.assign(size_t(numRows) + 1u, 0);
	matrixOut->colIndices.clear();
	matrixOut->values.clear();

	if (numRows <= 0 || numCols <= 0) {
		return;
	}

	triplets.erase(
		std::remove_if(
			triplets.begin(),
			triplets.end(),
			[numRows, numCols](SparseTriplet t) {
				return
					t.value == 0.0 ||
					t.row < 0 ||
					t.row >= numRows ||
					t.col < 0 ||
					t.col >= numCols;
			}),
		triplets.end());

	std::sort(
		triplets.begin(),
		triplets.end(),
		[](SparseTriplet a, SparseTriplet b) {
			return a.row < b.row || (a.row == b.row && a.col < b.col);
		});

	for (size_t i = 0; i < triplets.size();) {
		const i32 row = triplets[i].row;
		const i32 col = triplets[i].col;
		f64 value = 0.0;
		do {
			value += triplets[i].value;
			++i;
		} while (i < triplets.size() && triplets[i].row == row && triplets[i].col == col);

		if (value == 0.0) {
			continue;
		}
		++matrixOut->rowOffsets[size_t(row) + 1u];
		matrixOut->colIndices.push_back(col);
		matrixOut->values.push_back(value);
	}

	for (i32 row = 0; row < numRows; ++row) {
		matrixOut->rowOffsets[size_t(row) + 1u] += matrixOut->rowOffsets[size_t(row)];
	}
}

inline void matVecCsr(std::vector<f64> *yOut, const CsrMatrix &a, const std::vector<f64> &x) {
	if (yOut == nullptr || i32(x.size()) != a.numCols) {
		return;
	}

	yOut->assign(size_t(a.numRows), 0.0);
	for (i32 row = 0; row < a.numRows; ++row) {
		f64 sum = 0.0;
		const i32 begin = a.rowOffsets[size_t(row)];
		const i32 end = a.rowOffsets[size_t(row) + 1u];
		for (i32 idx = begin; idx < end; ++idx) {
			sum += a.values[size_t(idx)]*x[size_t(a.colIndices[size_t(idx)])];
		}
		(*yOut)[size_t(row)] = sum;
	}
}

inline void computeCsrDiagonal(std::vector<f64> *diagOut, const CsrMatrix &a, f64 minDiag = 0.0) {
	if (diagOut == nullptr) {
		return;
	}

	diagOut->assign(size_t(a.numRows), minDiag);
	for (i32 row = 0; row < a.numRows; ++row) {
		const i32 begin = a.rowOffsets[size_t(row)];
		const i32 end = a.rowOffsets[size_t(row) + 1u];
		for (i32 idx = begin; idx < end; ++idx) {
			if (a.colIndices[size_t(idx)] == row) {
				(*diagOut)[size_t(row)] = std::max(std::fabs(a.values[size_t(idx)]), minDiag);
				break;
			}
		}
	}
}

inline void addScaledDiagonalToCsr(CsrMatrix *a, const std::vector<f64> &diag, f64 scale) {
	if (a == nullptr || i32(diag.size()) != a->numRows || a->numRows != a->numCols) {
		return;
	}

	std::vector<SparseTriplet> triplets;
	triplets.reserve(a->values.size() + diag.size());
	for (i32 row = 0; row < a->numRows; ++row) {
		const i32 begin = a->rowOffsets[size_t(row)];
		const i32 end = a->rowOffsets[size_t(row) + 1u];
		for (i32 idx = begin; idx < end; ++idx) {
			SparseTriplet t = {};
			t.row = row;
			t.col = a->colIndices[size_t(idx)];
			t.value = a->values[size_t(idx)];
			triplets.push_back(t);
		}

		SparseTriplet d = {};
		d.row = row;
		d.col = row;
		d.value = scale*diag[size_t(row)];
		triplets.push_back(d);
	}

	compressTripletsToCsr(a, std::move(triplets), a->numRows, a->numCols);
}

inline f64 dotVector(const std::vector<f64> &a, const std::vector<f64> &b) {
	const i32 n = std::min(i32(a.size()), i32(b.size()));
	f64 result = 0.0;
	for (i32 i = 0; i < n; ++i) {
		result += a[size_t(i)]*b[size_t(i)];
	}
	return result;
}

inline f64 normVector(const std::vector<f64> &x) {
	return std::sqrt(std::max(0.0, dotVector(x, x)));
}

inline bool solvePcg(
	std::vector<f64> *xOut,
	PcgResult *resultOut,
	const CsrMatrix &a,
	const std::vector<f64> &b,
	const std::vector<f64> &preconditionerDiag,
	PcgParams params) {
	if (resultOut != nullptr) {
		*resultOut = {};
	}
	if (xOut == nullptr || a.numRows <= 0 || a.numRows != a.numCols || i32(b.size()) != a.numRows) {
		return false;
	}

	std::vector<f64> x(size_t(a.numRows), 0.0);
	std::vector<f64> r = b;
	std::vector<f64> z(size_t(a.numRows), 0.0);
	std::vector<f64> p(size_t(a.numRows), 0.0);
	std::vector<f64> ap;

	for (i32 i = 0; i < a.numRows; ++i) {
		const f64 diag = (i < i32(preconditionerDiag.size())) ? preconditionerDiag[size_t(i)] : 1.0;
		z[size_t(i)] = r[size_t(i)]/(std::fabs(diag) > 1.0e-30 ? diag : 1.0);
		p[size_t(i)] = z[size_t(i)];
	}

	f64 rz = dotVector(r, z);
	const f64 initialNorm = normVector(r);
	const f64 targetNorm = std::max(params.absoluteTolerance, params.relativeTolerance*initialNorm);
	if (resultOut != nullptr) {
		resultOut->initialResidualNorm = initialNorm;
		resultOut->finalResidualNorm = initialNorm;
	}

	if (initialNorm <= targetNorm) {
		if (resultOut != nullptr) {
			resultOut->converged = true;
		}
		*xOut = std::move(x);
		return true;
	}

	for (i32 iter = 0; iter < params.maxIterations; ++iter) {
		matVecCsr(&ap, a, p);
		const f64 denom = dotVector(p, ap);
		if (std::fabs(denom) <= params.denominatorEps) {
			break;
		}

		const f64 alpha = rz/denom;
		for (i32 i = 0; i < a.numRows; ++i) {
			x[size_t(i)] += alpha*p[size_t(i)];
			r[size_t(i)] -= alpha*ap[size_t(i)];
		}

		const f64 residualNorm = normVector(r);
		if (resultOut != nullptr) {
			resultOut->iterations = iter + 1;
			resultOut->finalResidualNorm = residualNorm;
		}
		if (residualNorm <= targetNorm) {
			if (resultOut != nullptr) {
				resultOut->converged = true;
			}
			*xOut = std::move(x);
			return true;
		}

		for (i32 i = 0; i < a.numRows; ++i) {
			const f64 diag = (i < i32(preconditionerDiag.size())) ? preconditionerDiag[size_t(i)] : 1.0;
			z[size_t(i)] = r[size_t(i)]/(std::fabs(diag) > 1.0e-30 ? diag : 1.0);
		}

		const f64 nextRz = dotVector(r, z);
		const f64 beta = nextRz/(std::fabs(rz) > 1.0e-30 ? rz : 1.0);
		for (i32 i = 0; i < a.numRows; ++i) {
			p[size_t(i)] = z[size_t(i)] + beta*p[size_t(i)];
		}
		rz = nextRz;
	}

	*xOut = std::move(x);
	return resultOut == nullptr || resultOut->iterations > 0;
}

inline bool solveSparseLmStep(
	std::vector<f64> *deltaOut,
	SparseLmStepResult *resultOut,
	const CsrMatrix &jtJ,
	const std::vector<f64> &jtR,
	SparseLmStepParams params) {
	if (resultOut != nullptr) {
		*resultOut = {};
		resultOut->lambda = params.lambda;
	}
	if (deltaOut == nullptr || jtJ.numRows <= 0 || jtJ.numRows != jtJ.numCols || i32(jtR.size()) != jtJ.numRows) {
		return false;
	}

	params.lambda = std::max(params.minLambda, std::min(params.lambda, params.maxLambda));

	std::vector<f64> diag;
	computeCsrDiagonal(&diag, jtJ, params.minDiag);
	CsrMatrix a = jtJ;
	addScaledDiagonalToCsr(&a, diag, params.lambda);

	std::vector<f64> preconditioner;
	computeCsrDiagonal(&preconditioner, a, params.minDiag);

	std::vector<f64> rhs(jtR.size(), 0.0);
	for (i32 i = 0; i < i32(jtR.size()); ++i) {
		rhs[size_t(i)] = -jtR[size_t(i)];
	}

	PcgResult pcg = {};
	std::vector<f64> delta;
	if (!solvePcg(&delta, &pcg, a, rhs, preconditioner, params.pcg)) {
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
		resultOut->converged = pcg.converged;
		resultOut->iterations = pcg.iterations;
		resultOut->lambda = params.lambda;
		resultOut->maxBlockNorm = clamped ? params.maxBlockNorm : beforeClampNorm;
		resultOut->initialResidualNorm = pcg.initialResidualNorm;
		resultOut->finalResidualNorm = pcg.finalResidualNorm;
	}
	*deltaOut = std::move(delta);
	return true;
}

inline void decreaseSparseLmLambda(f64 *lambda, const SparseLmStepParams &params) {
	if (lambda == nullptr) {
		return;
	}

	*lambda = std::max(params.minLambda, (*lambda)*params.lambdaDecrease);
}

inline void increaseSparseLmLambda(f64 *lambda, const SparseLmStepParams &params) {
	if (lambda == nullptr) {
		return;
	}

	*lambda = std::min(params.maxLambda, (*lambda)*params.lambdaIncrease);
}

} // namespace numeric
