#pragma once

/**
 * @file experiment_diagnostics.cuh
 * @brief Shared run logging and plot output for thesis experiments.
 */

#include "spr_global_include.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace expdiag {

struct IterationSample {
	i32 iteration = 0;
	f64 elapsedMs = 0.0;
	f64 energy = 0.0;
	f64 maxViolation = 0.0;
	f64 avgViolation = 0.0;
	f64 minPhi = 0.0;
	f64 rmsOffset = 0.0;
	f64 rmsNormal = 0.0;
	f64 rmsLandmark = 0.0;
	f64 rmsGuide = 0.0;
	f64 maxControl = 0.0;
	i32 numViolating = 0;
};

struct RunDiagnostics {
	const char *name = "";
	std::chrono::steady_clock::time_point startTime = {};
	f64 totalElapsedMs = 0.0;
	i32 stopIteration = -1;
	std::string stopReason;
	std::vector<IterationSample> samples;
};

struct StopCriteria {
	i32 minIterations = 20;
	i32 patience = 16;
	i32 minFeasibleSamples = 17;
	f64 relativeEnergyWindow = 1.0e-3;
	f64 absoluteEnergyWindow = 1.0e-6;
	f64 maxViolation = 1.0e-4;
	i32 maxViolating = 0;
};

struct StopStatus {
	bool shouldStop = false;
	i32 iteration = 0;
	i32 feasibleSamples = 0;
	f64 windowDecrease = 0.0;
	f64 threshold = 0.0;
	char reason[192] = {};
};

inline f64 elapsedMsSince(std::chrono::steady_clock::time_point startTime) {
	using Duration = std::chrono::duration<f64, std::milli>;
	return Duration(std::chrono::steady_clock::now() - startTime).count();
}

inline void beginRun(RunDiagnostics *run, const char *name) {
	if (run == nullptr) {
		return;
	}

	run->name = name != nullptr ? name : "";
	run->samples.clear();
	run->totalElapsedMs = 0.0;
	run->stopIteration = -1;
	run->stopReason.clear();
	run->startTime = std::chrono::steady_clock::now();
}

inline void endRun(RunDiagnostics *run) {
	if (run == nullptr) {
		return;
	}
	run->totalElapsedMs = elapsedMsSince(run->startTime);
}

inline void appendSample(RunDiagnostics *run, IterationSample sample) {
	if (run == nullptr) {
		return;
	}

	sample.elapsedMs = elapsedMsSince(run->startTime);
	run->samples.push_back(sample);
}

inline i32 nextIteration(const RunDiagnostics &run) {
	return run.samples.empty() ? 0 : run.samples.back().iteration + 1;
}

inline f64 finiteOrZero(f64 x) {
	return std::isfinite(x) ? x : 0.0;
}

inline bool stopSampleFeasible(const IterationSample &sample, const StopCriteria &criteria) {
	return finiteOrZero(sample.maxViolation) <= criteria.maxViolation &&
		sample.numViolating <= criteria.maxViolating;
}

inline i32 countTrailingFeasibleSamples(const RunDiagnostics &run, const StopCriteria &criteria) {
	i32 count = 0;
	for (i32 i = i32(run.samples.size()) - 1; i >= 0; --i) {
		if (!stopSampleFeasible(run.samples[size_t(i)], criteria)) {
			break;
		}
		++count;
	}
	return count;
}

inline StopStatus evaluateStop(const RunDiagnostics &run, const StopCriteria &criteria) {
	StopStatus status = {};
	if (run.samples.empty()) {
		return status;
	}

	const IterationSample &last = run.samples.back();
	status.iteration = last.iteration;
	if (last.iteration < criteria.minIterations || !stopSampleFeasible(last, criteria)) {
		return status;
	}

	const i32 requiredFeasible = std::max(criteria.minFeasibleSamples, criteria.patience + 1);
	status.feasibleSamples = countTrailingFeasibleSamples(run, criteria);
	if (status.feasibleSamples < requiredFeasible || i32(run.samples.size()) <= criteria.patience) {
		return status;
	}

	const i32 baseIndex = i32(run.samples.size()) - 1 - criteria.patience;
	const IterationSample &base = run.samples[size_t(baseIndex)];
	const f64 initialScale = std::max(1.0, std::fabs(finiteOrZero(run.samples.front().energy)));
	status.windowDecrease = finiteOrZero(base.energy) - finiteOrZero(last.energy);
	status.threshold = std::max(criteria.absoluteEnergyWindow, criteria.relativeEnergyWindow*initialScale);
	if (status.windowDecrease <= status.threshold) {
		status.shouldStop = true;
		std::snprintf(
			status.reason,
			sizeof(status.reason),
			"converged at iteration %d: feasible for %d samples, dE(%d)=%g <= %g",
			status.iteration,
			status.feasibleSamples,
			criteria.patience,
			status.windowDecrease,
			status.threshold);
	}
	return status;
}

inline void recordStop(RunDiagnostics *run, const StopStatus &status) {
	if (run == nullptr || !status.shouldStop) {
		return;
	}
	run->stopIteration = status.iteration;
	run->stopReason = status.reason;
}

inline bool recordStopIfReached(RunDiagnostics *run, const StopCriteria &criteria) {
	if (run == nullptr) {
		return false;
	}
	const StopStatus status = evaluateStop(*run, criteria);
	if (!status.shouldStop) {
		return false;
	}
	recordStop(run, status);
	std::printf("%s\n", status.reason);
	return true;
}

inline bool writeCsv(const char *path, const RunDiagnostics &run) {
	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	std::fprintf(
		file,
		"iteration,elapsed_ms,energy,max_violation,avg_violation,min_phi,rms_offset,rms_normal,rms_landmark,rms_guide,max_control,num_violating\n");
	for (const IterationSample &s : run.samples) {
		std::fprintf(
			file,
			"%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
			s.iteration,
			s.elapsedMs,
			finiteOrZero(s.energy),
			finiteOrZero(s.maxViolation),
			finiteOrZero(s.avgViolation),
			finiteOrZero(s.minPhi),
			finiteOrZero(s.rmsOffset),
			finiteOrZero(s.rmsNormal),
			finiteOrZero(s.rmsLandmark),
			finiteOrZero(s.rmsGuide),
			finiteOrZero(s.maxControl),
			s.numViolating);
	}

	std::fclose(file);
	return true;
}

inline f64 maxAbsMetric(const RunDiagnostics &run, i32 metric) {
	f64 result = 0.0;
	for (const IterationSample &s : run.samples) {
		f64 value = 0.0;
		if (metric == 0) {
			value = s.energy;
		} else if (metric == 1) {
			value = s.maxViolation;
		} else if (metric == 2) {
			value = f64(s.numViolating);
		}
		if (std::isfinite(value)) {
			result = std::max(result, std::fabs(value));
		}
	}
	return result > 0.0 ? result : 1.0;
}

inline size_t focusedPlotSampleCount(const RunDiagnostics &run) {
	const size_t n = run.samples.size();
	if (n == 0) {
		return 0;
	}

	const f64 energyScale = maxAbsMetric(run, 0);
	const f64 violationScale = maxAbsMetric(run, 1);
	const f64 countScale = maxAbsMetric(run, 2);
	const f64 firstEnergy = finiteOrZero(run.samples.front().energy)/energyScale;
	const f64 finalEnergy = finiteOrZero(run.samples.back().energy)/energyScale;
	const f64 energyBand = std::max(1.0e-6, 0.05*std::fabs(firstEnergy - finalEnergy));
	const f64 violationEps = std::max(1.0e-8, 1.0e-2*violationScale);
	const i32 countEps = std::max<i32>(1, i32(std::ceil(0.01*countScale)));

	size_t lastActive = 0;
	for (size_t i = 0; i < n; ++i) {
		const IterationSample &s = run.samples[i];
		const f64 normalizedEnergy = finiteOrZero(s.energy)/energyScale;
		const bool clearanceActive = finiteOrZero(s.maxViolation) > violationEps || s.numViolating > countEps;
		const bool energyActive = std::fabs(normalizedEnergy - finalEnergy) > energyBand;
		if (clearanceActive || energyActive) {
			lastActive = i;
		}
	}

	const size_t tailSamples = 8;
	const size_t minSamples = std::min<size_t>(n, 12);
	size_t focused = std::min(n, lastActive + 1 + tailSamples);
	focused = std::max(focused, minSamples);
	return focused;
}

inline bool writeSummary(const char *path, const RunDiagnostics &run) {
	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}

	std::fprintf(file, "name,%s\n", run.name);
	std::fprintf(file, "samples,%zu\n", run.samples.size());
	std::fprintf(file, "total_elapsed_ms,%.9g\n", run.totalElapsedMs);
	if (run.stopIteration >= 0) {
		std::fprintf(file, "stop_iteration,%d\n", run.stopIteration);
		std::fprintf(file, "stop_reason,%s\n", run.stopReason.c_str());
	}
	if (!run.samples.empty()) {
		const IterationSample &first = run.samples.front();
		const IterationSample &last = run.samples.back();
		std::fprintf(file, "first_iteration,%d\n", first.iteration);
		std::fprintf(file, "last_iteration,%d\n", last.iteration);
		std::fprintf(file, "initial_energy,%.9g\n", first.energy);
		std::fprintf(file, "final_energy,%.9g\n", last.energy);
		std::fprintf(file, "initial_max_violation,%.9g\n", first.maxViolation);
		std::fprintf(file, "final_max_violation,%.9g\n", last.maxViolation);
		std::fprintf(file, "initial_num_violating,%d\n", first.numViolating);
		std::fprintf(file, "final_num_violating,%d\n", last.numViolating);
	}

	std::fclose(file);
	return true;
}

struct PlotRgb {
	u8 r;
	u8 g;
	u8 b;
};

inline u8 plotGlyphRow(char c, i32 row) {
	if (c >= 'a' && c <= 'z') {
		c = char(c - 'a' + 'A');
	}

	switch (c) {
	case 'A': { static const u8 r[7] = {14, 17, 17, 31, 17, 17, 17}; return r[row]; }
	case 'B': { static const u8 r[7] = {30, 17, 17, 30, 17, 17, 30}; return r[row]; }
	case 'C': { static const u8 r[7] = {15, 16, 16, 16, 16, 16, 15}; return r[row]; }
	case 'D': { static const u8 r[7] = {30, 17, 17, 17, 17, 17, 30}; return r[row]; }
	case 'E': { static const u8 r[7] = {31, 16, 16, 30, 16, 16, 31}; return r[row]; }
	case 'F': { static const u8 r[7] = {31, 16, 16, 30, 16, 16, 16}; return r[row]; }
	case 'G': { static const u8 r[7] = {15, 16, 16, 23, 17, 17, 15}; return r[row]; }
	case 'H': { static const u8 r[7] = {17, 17, 17, 31, 17, 17, 17}; return r[row]; }
	case 'I': { static const u8 r[7] = {31, 4, 4, 4, 4, 4, 31}; return r[row]; }
	case 'J': { static const u8 r[7] = {7, 2, 2, 2, 18, 18, 12}; return r[row]; }
	case 'K': { static const u8 r[7] = {17, 18, 20, 24, 20, 18, 17}; return r[row]; }
	case 'L': { static const u8 r[7] = {16, 16, 16, 16, 16, 16, 31}; return r[row]; }
	case 'M': { static const u8 r[7] = {17, 27, 21, 21, 17, 17, 17}; return r[row]; }
	case 'N': { static const u8 r[7] = {17, 25, 21, 19, 17, 17, 17}; return r[row]; }
	case 'O': { static const u8 r[7] = {14, 17, 17, 17, 17, 17, 14}; return r[row]; }
	case 'P': { static const u8 r[7] = {30, 17, 17, 30, 16, 16, 16}; return r[row]; }
	case 'Q': { static const u8 r[7] = {14, 17, 17, 17, 21, 18, 13}; return r[row]; }
	case 'R': { static const u8 r[7] = {30, 17, 17, 30, 20, 18, 17}; return r[row]; }
	case 'S': { static const u8 r[7] = {15, 16, 16, 14, 1, 1, 30}; return r[row]; }
	case 'T': { static const u8 r[7] = {31, 4, 4, 4, 4, 4, 4}; return r[row]; }
	case 'U': { static const u8 r[7] = {17, 17, 17, 17, 17, 17, 14}; return r[row]; }
	case 'V': { static const u8 r[7] = {17, 17, 17, 17, 10, 10, 4}; return r[row]; }
	case 'W': { static const u8 r[7] = {17, 17, 17, 21, 21, 27, 17}; return r[row]; }
	case 'X': { static const u8 r[7] = {17, 10, 4, 4, 4, 10, 17}; return r[row]; }
	case 'Y': { static const u8 r[7] = {17, 10, 4, 4, 4, 4, 4}; return r[row]; }
	case 'Z': { static const u8 r[7] = {31, 1, 2, 4, 8, 16, 31}; return r[row]; }
	case '0': { static const u8 r[7] = {14, 17, 19, 21, 25, 17, 14}; return r[row]; }
	case '1': { static const u8 r[7] = {4, 12, 4, 4, 4, 4, 14}; return r[row]; }
	case '2': { static const u8 r[7] = {14, 17, 1, 2, 4, 8, 31}; return r[row]; }
	case '3': { static const u8 r[7] = {30, 1, 1, 14, 1, 1, 30}; return r[row]; }
	case '4': { static const u8 r[7] = {2, 6, 10, 18, 31, 2, 2}; return r[row]; }
	case '5': { static const u8 r[7] = {31, 16, 30, 1, 1, 17, 14}; return r[row]; }
	case '6': { static const u8 r[7] = {6, 8, 16, 30, 17, 17, 14}; return r[row]; }
	case '7': { static const u8 r[7] = {31, 1, 2, 4, 8, 8, 8}; return r[row]; }
	case '8': { static const u8 r[7] = {14, 17, 17, 14, 17, 17, 14}; return r[row]; }
	case '9': { static const u8 r[7] = {14, 17, 17, 15, 1, 2, 28}; return r[row]; }
	case '.': { static const u8 r[7] = {0, 0, 0, 0, 0, 12, 12}; return r[row]; }
	case ':': { static const u8 r[7] = {0, 12, 12, 0, 12, 12, 0}; return r[row]; }
	case '-': { static const u8 r[7] = {0, 0, 0, 31, 0, 0, 0}; return r[row]; }
	case '/': { static const u8 r[7] = {1, 1, 2, 4, 8, 16, 16}; return r[row]; }
	default: return 0;
	}
}

inline void plotSetPixel(
	std::vector<u8> *pixels,
	i32 width,
	i32 height,
	i32 x,
	i32 y,
	PlotRgb color) {
	if (pixels == nullptr || x < 0 || x >= width || y < 0 || y >= height) {
		return;
	}

	const size_t idx = (size_t(y)*size_t(width) + size_t(x))*3u;
	(*pixels)[idx + 0] = color.r;
	(*pixels)[idx + 1] = color.g;
	(*pixels)[idx + 2] = color.b;
}

inline void plotFillRect(
	std::vector<u8> *pixels,
	i32 width,
	i32 height,
	i32 x0,
	i32 y0,
	i32 x1,
	i32 y1,
	PlotRgb color) {
	if (x0 > x1) { std::swap(x0, x1); }
	if (y0 > y1) { std::swap(y0, y1); }
	for (i32 y = y0; y <= y1; ++y) {
		for (i32 x = x0; x <= x1; ++x) {
			plotSetPixel(pixels, width, height, x, y, color);
		}
	}
}

inline void plotDrawLine(
	std::vector<u8> *pixels,
	i32 width,
	i32 height,
	i32 x0,
	i32 y0,
	i32 x1,
	i32 y1,
	PlotRgb color) {
	const i32 dx = std::abs(x1 - x0);
	const i32 sx = x0 < x1 ? 1 : -1;
	const i32 dy = -std::abs(y1 - y0);
	const i32 sy = y0 < y1 ? 1 : -1;
	i32 err = dx + dy;

	for (;;) {
		plotSetPixel(pixels, width, height, x0, y0, color);
		if (x0 == x1 && y0 == y1) {
			break;
		}
		const i32 e2 = 2*err;
		if (e2 >= dy) {
			err += dy;
			x0 += sx;
		}
		if (e2 <= dx) {
			err += dx;
			y0 += sy;
		}
	}
}

inline void plotDrawText(
	std::vector<u8> *pixels,
	i32 width,
	i32 height,
	i32 x,
	i32 y,
	const std::string &text,
	PlotRgb color,
	i32 scale = 1) {
	scale = std::max(scale, 1);
	i32 cursor = x;
	for (char c : text) {
		for (i32 row = 0; row < 7; ++row) {
			const u8 bits = plotGlyphRow(c, row);
			for (i32 col = 0; col < 5; ++col) {
				if ((bits & (1u << (4 - col))) == 0) {
					continue;
				}
				plotFillRect(
					pixels,
					width,
					height,
					cursor + col*scale,
					y + row*scale,
					cursor + (col + 1)*scale - 1,
					y + (row + 1)*scale - 1,
					color);
			}
		}
		cursor += 6*scale;
	}
}

inline f64 clamp01(f64 x) {
	if (!std::isfinite(x)) {
		return 0.0;
	}
	return std::max(0.0, std::min(1.0, x));
}

inline f64 samplePlotMetric(const IterationSample &s, i32 metric) {
	if (metric == 0) {
		return s.energy;
	}
	if (metric == 1) {
		return s.maxViolation;
	}
	return f64(s.numViolating);
}

inline bool writePlotPpm(const char *path, const RunDiagnostics &run) {
	if (path == nullptr || run.samples.empty()) {
		return false;
	}

	constexpr i32 width = 960;
	constexpr i32 height = 560;
	constexpr i32 left = 74;
	constexpr i32 top = 86;
	constexpr i32 right = 34;
	constexpr i32 bottom = 58;
	constexpr i32 plotW = width - left - right;
	constexpr i32 plotH = height - top - bottom;
	const PlotRgb bg = {248, 248, 246};
	const PlotRgb grid = {221, 223, 224};
	const PlotRgb axis = {45, 48, 52};
	const PlotRgb text = {32, 34, 38};
	const PlotRgb energyColor = {31, 75, 190};
	const PlotRgb violationColor = {204, 49, 35};
	const PlotRgb countColor = {45, 139, 72};

	std::vector<u8> pixels(size_t(width)*size_t(height)*3u, 255);
	plotFillRect(&pixels, width, height, 0, 0, width - 1, height - 1, bg);

	plotDrawText(&pixels, width, height, 18, 16, run.name, text, 2);
	plotDrawText(&pixels, width, height, 18, 48, "NORMALIZED RUN DIAGNOSTICS", text, 1);

	const i32 legendX = 610;
	const i32 legendY = 18;
	plotDrawLine(&pixels, width, height, legendX, legendY + 5, legendX + 32, legendY + 5, energyColor);
	plotDrawText(&pixels, width, height, legendX + 42, legendY, "ENERGY", text, 1);
	plotDrawLine(&pixels, width, height, legendX, legendY + 23, legendX + 32, legendY + 23, violationColor);
	plotDrawText(&pixels, width, height, legendX + 42, legendY + 18, "MAX VIOLATION", text, 1);
	plotDrawLine(&pixels, width, height, legendX, legendY + 41, legendX + 32, legendY + 41, countColor);
	plotDrawText(&pixels, width, height, legendX + 42, legendY + 36, "VIOLATING VERTICES", text, 1);

	for (i32 i = 0; i <= 4; ++i) {
		const i32 y = top + (plotH*i)/4;
		plotDrawLine(&pixels, width, height, left, y, left + plotW, y, grid);
	}
	for (i32 i = 0; i <= 8; ++i) {
		const i32 x = left + (plotW*i)/8;
		plotDrawLine(&pixels, width, height, x, top, x, top + plotH, grid);
	}
	plotDrawLine(&pixels, width, height, left, top, left + plotW, top, axis);
	plotDrawLine(&pixels, width, height, left, top + plotH, left + plotW, top + plotH, axis);
	plotDrawLine(&pixels, width, height, left, top, left, top + plotH, axis);
	plotDrawLine(&pixels, width, height, left + plotW, top, left + plotW, top + plotH, axis);
	plotDrawText(&pixels, width, height, left + plotW/2 - 52, height - 32, "ITERATION", text, 1);
	plotDrawText(&pixels, width, height, 8, top + 4, "1.0", text, 1);
	plotDrawText(&pixels, width, height, 8, top + plotH - 8, "0.0", text, 1);

	const size_t plotCount = focusedPlotSampleCount(run);
	const i32 xMin = run.samples.front().iteration;
	const i32 xMax = std::max(xMin + 1, run.samples[plotCount - 1].iteration);
	const f64 scales[3] = {
		maxAbsMetric(run, 0),
		maxAbsMetric(run, 1),
		maxAbsMetric(run, 2),
	};
	const PlotRgb colors[3] = {energyColor, violationColor, countColor};

	for (i32 metric = 0; metric < 3; ++metric) {
		bool hasPrev = false;
		i32 prevX = 0;
		i32 prevY = 0;
		for (size_t i = 0; i < plotCount; ++i) {
			const IterationSample &s = run.samples[i];
			const f64 x01 = f64(s.iteration - xMin)/f64(xMax - xMin);
			const f64 y01 = clamp01(samplePlotMetric(s, metric)/scales[metric]);
			const i32 x = left + i32(std::round(clamp01(x01)*f64(plotW)));
			const i32 y = top + plotH - i32(std::round(y01*f64(plotH)));
			if (hasPrev) {
				plotDrawLine(&pixels, width, height, prevX, prevY, x, y, colors[metric]);
				plotDrawLine(&pixels, width, height, prevX, prevY + 1, x, y + 1, colors[metric]);
			}
			prevX = x;
			prevY = y;
			hasPrev = true;
		}
	}

	char wallText[96];
	std::snprintf(wallText, sizeof(wallText), "WALL %.1f MS  SHOWN %zu/%zu", run.totalElapsedMs, plotCount, run.samples.size());
	plotDrawText(&pixels, width, height, width - 300, height - 32, wallText, text, 1);

	FILE *file = std::fopen(path, "wb");
	if (file == nullptr) {
		return false;
	}
	std::fprintf(file, "P6\n%d %d\n255\n", width, height);
	std::fwrite(pixels.data(), 1, pixels.size(), file);
	std::fclose(file);
	return true;
}

inline bool writeRunArtifacts(const char *prefix, RunDiagnostics *run) {
	if (prefix == nullptr || run == nullptr) {
		return false;
	}

	endRun(run);
	const std::string base(prefix);
	const bool csvOk = writeCsv((base + "_diagnostics.csv").c_str(), *run);
	const bool summaryOk = writeSummary((base + "_summary.csv").c_str(), *run);
	const bool ppmOk = writePlotPpm((base + "_diagnostics.ppm").c_str(), *run);
	return csvOk && summaryOk && ppmOk;
}

} // namespace expdiag
