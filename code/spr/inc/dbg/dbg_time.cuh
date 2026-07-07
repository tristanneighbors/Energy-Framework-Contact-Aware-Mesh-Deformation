#pragma once

/**
 * @file dbg_time.cuh
 * @brief Nested CPU and GPU timing reports.
 *
 * CPU timers use `std::chrono::steady_clock`. GPU timers use CUDA events on a
 * root stream and synchronize when the root timer ends. Both retain the last
 * completed report until the next root timer begins, and both track inclusive
 * elapsed time plus child elapsed time so callers can compute self time.
 *
 * GPU sub-timers should stay on the parent stream. Use the no-stream overload
 * to inherit the parent stream, or pass the stream explicitly when that makes a
 * call site clearer.
 */

#include "cuda_utils.cuh"
#include "dbg_config.hpp"
#include "dbg_log.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

//==============================================================================
// CPU timers
//==============================================================================

namespace dbg {

struct CpuTimerRecord {
	std::string name;

	int parentIdx = INVALID_IDX;
	int depth = 0;

	double startOffset_ms = 0.0;
	double elapsed_ms = 0.0;
	double childElapsed_ms = 0.0;
};

struct CpuTimerFrame {
	int recordIdx = INVALID_IDX;
	std::chrono::steady_clock::time_point startTime = {};
};

/** Per-thread CPU timer state. Prefer `getCpuTimerState()` over owning this directly. */
struct CpuTimerState {
	bool isActive = false;
	bool hasReport = false;

	std::chrono::steady_clock::time_point rootStartTime = {};

	std::vector<CpuTimerRecord> records;
	std::vector<CpuTimerFrame> stack;
};

CpuTimerState *getCpuTimerState();

void reserveTimerRecords(int capacity);
void clearTimerReport();

const CpuTimerState *getLastTimerReport();

/** Start a root CPU timing report, clearing any prior active stack. */
void beginTimer(const char *name = "root");
void endTimer();

/** Start and end a nested CPU timer under the active timer frame. */
void beginSubTimer(const char *name);
void endSubTimer();

[[nodiscard]] bool isTimerActive();
[[nodiscard]] int getTimerDepth();

void printTimerReport(
	FILE *file,
	const CpuTimerState *timer);

void logTimerReport(
	const CpuTimerState *timer,
	LogDest dest = LogDest::File);

} // namespace dbg

//==============================================================================
// CPU timer macros
//==============================================================================

#if DEV_ENABLE_CPU_TIMER

#define DBG_BEGIN_TIMER(name) \
	dbg::beginTimer(name)

#define DBG_END_TIMER() \
	dbg::endTimer()

#define DBG_BEGIN_SUB_TIMER(name) \
	dbg::beginSubTimer(name)

#define DBG_END_SUB_TIMER() \
	dbg::endSubTimer()

#define DBG_LOG_TIMER_REPORT() \
	dbg::logTimerReport(dbg::getLastTimerReport())

#else

#define DBG_BEGIN_TIMER(name) ((void)0)
#define DBG_END_TIMER() ((void)0)
#define DBG_BEGIN_SUB_TIMER(name) ((void)0)
#define DBG_END_SUB_TIMER() ((void)0)
#define DBG_LOG_TIMER_REPORT() ((void)0)

#endif

//==============================================================================
// GPU timers
//==============================================================================

namespace dbg {

struct GpuTimerRecord {
	std::string name;

	int parentIdx = INVALID_IDX;
	int depth = 0;

	cudaStream_t stream = nullptr;

	cudaEvent_t startEvent = nullptr;
	cudaEvent_t stopEvent = nullptr;

	float startOffset_ms = 0.0f;
	float elapsed_ms = 0.0f;
	float childElapsed_ms = 0.0f;
};

/** GPU timer state; stack-allocated instances must be initialized with `initGpuTimer()`. */
struct GpuTimerState {
	bool isInit = false;
	bool isActive = false;
	bool hasReport = false;

	int device = INVALID_IDX;

	std::vector<GpuTimerRecord> records;
	std::vector<int> stack;
};

GpuTimerState *getGpuTimerState();

void initGpuTimer(
	GpuTimerState *timer,
	int reserveRecordCapacity = DEV_TIMER_DEFAULT_RECORD_CAPACITY);

void destroyGpuTimer(GpuTimerState *timer);

void reserveGpuTimerRecords(int capacity);
void clearGpuTimerReport();

const GpuTimerState *getLastGpuTimerReport();

/** Start a root GPU timing report on `stream`. Ending the root synchronizes its events. */
void beginGpuTimer(
	const char *name = "gpu",
	cudaStream_t stream = 0);

void endGpuTimer();

void beginGpuSubTimer(
	const char *name,
	cudaStream_t stream);

/** Start a nested GPU timer on the parent timer's stream. */
void beginGpuSubTimer(const char *name);

void endGpuSubTimer();

[[nodiscard]] bool isGpuTimerActive();
[[nodiscard]] int getGpuTimerDepth();

void printGpuTimerReport(
	FILE *file,
	const GpuTimerState *timer);

void logGpuTimerReport(
	const GpuTimerState *timer,
	LogDest dest = LogDest::File);

} // namespace dbg

//==============================================================================
// GPU timer macros
//==============================================================================

#if DEV_ENABLE_GPU_TIMER

#define DBG_BEGIN_GPU_TIMER(...) \
	dbg::beginGpuTimer(__VA_ARGS__)

#define DBG_END_GPU_TIMER() \
	dbg::endGpuTimer()

#define DBG_BEGIN_GPU_SUB_TIMER(...) \
	dbg::beginGpuSubTimer(__VA_ARGS__)

#define DBG_END_GPU_SUB_TIMER() \
	dbg::endGpuSubTimer()

#define DBG_LOG_GPU_TIMER_REPORT() \
	dbg::logGpuTimerReport(dbg::getLastGpuTimerReport())

#else

#define DBG_BEGIN_GPU_TIMER(...) ((void)0)
#define DBG_END_GPU_TIMER() ((void)0)
#define DBG_BEGIN_GPU_SUB_TIMER(...) ((void)0)
#define DBG_END_GPU_SUB_TIMER() ((void)0)
#define DBG_LOG_GPU_TIMER_REPORT() ((void)0)

#endif
