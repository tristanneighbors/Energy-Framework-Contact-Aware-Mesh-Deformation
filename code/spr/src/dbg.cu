#include "dbg.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

//==============================================================================
// File-local helpers
//==============================================================================

static constexpr int THREAD_NAME_CAPACITY = 128;

static const char *safeText(const char *text, const char *fallback) {
	return text != nullptr ? text : fallback;
}

static double elapsedMs(
	std::chrono::steady_clock::time_point startTime,
	std::chrono::steady_clock::time_point endTime) {
	using Duration = std::chrono::duration<double, std::milli>;
	return Duration(endTime - startTime).count();
}

static bool logDestHasStdout(dbg::LogDest dest) {
	return dest == dbg::LogDest::Stdout || dest == dbg::LogDest::StdoutAndFile;
}

static bool logDestHasStderr(dbg::LogDest dest) {
	return dest == dbg::LogDest::Stderr || dest == dbg::LogDest::StderrAndFile;
}

static bool logDestHasFile(dbg::LogDest dest) {
	return
		dest == dbg::LogDest::File ||
		dest == dbg::LogDest::StdoutAndFile ||
		dest == dbg::LogDest::StderrAndFile;
}

static char *threadNameBuffer() {
	static thread_local char threadName[THREAD_NAME_CAPACITY] = {};
	return threadName;
}

static void appendIndent(std::string *out, int depth) {
	for (int i = 0; i < depth; ++i) {
		out->push_back('\t');
	}
}

static void appendCpuTimerReport(std::string *out, const dbg::CpuTimerState *timer) {
	if (timer == nullptr || !timer->hasReport) {
		out->append("no CPU timer report\n");
		return;
	}

	char line[1024];
	for (const dbg::CpuTimerRecord &record : timer->records) {
		const double self_ms = record.elapsed_ms - record.childElapsed_ms;
		appendIndent(out, record.depth);
		std::snprintf(
			line,
			sizeof(line),
			"%-32s %10.3f ms inclusive %10.3f ms self\n",
			record.name.c_str(),
			record.elapsed_ms,
			self_ms);
		out->append(line);
	}
}

static void appendGpuTimerReport(std::string *out, const dbg::GpuTimerState *timer) {
	if (timer == nullptr || !timer->hasReport) {
		out->append("no GPU timer report\n");
		return;
	}

	char line[1024];
	for (const dbg::GpuTimerRecord &record : timer->records) {
		const float self_ms = record.elapsed_ms - record.childElapsed_ms;
		appendIndent(out, record.depth);
		std::snprintf(
			line,
			sizeof(line),
			"%-32s %10.3f ms inclusive %10.3f ms self\n",
			record.name.c_str(),
			record.elapsed_ms,
			self_ms);
		out->append(line);
	}
}

static void makeLogPath(
	char *pathOut,
	size_t pathSize,
	const dbg::LogConfig *config) {
	const char *logDir = safeText(config->logDir, ".");
	const char *filePrefix = safeText(config->filePrefix, "dbg");
	const size_t logDirLen = std::strlen(logDir);
	const bool needsSlash =
		logDirLen > 0 &&
		logDir[logDirLen - 1] != '/' &&
		logDir[logDirLen - 1] != '\\';

	std::snprintf(
		pathOut,
		pathSize,
		needsSlash ? "%s/%s.log" : "%s%s.log",
		logDir,
		filePrefix);
}

static bool shouldOpenLogFile(const dbg::LogConfig *config) {
	return
		config->logDir != nullptr &&
		config->filePrefix != nullptr &&
		config->filePrefix[0] != '\0';
}

static void initLogUnlocked(dbg::LogState *state, const dbg::LogConfig *config) {
	if (state->file != nullptr) {
		std::fclose(state->file);
		state->file = nullptr;
	}

	state->config = config != nullptr ? *config : dbg::defaultLogConfig();
	state->filePath[0] = '\0';

	if (shouldOpenLogFile(&state->config)) {
		makeLogPath(state->filePath, sizeof(state->filePath), &state->config);
		const char *mode = state->config.shouldAppend ? "ab" : "wb";
		state->file = std::fopen(state->filePath, mode);

		if (state->file == nullptr) {
			std::fprintf(
				stderr,
				"dbg log: failed to open '%s'\n",
				state->filePath);
			std::fflush(stderr);
		}
	}

	state->isInit = true;
}

static void ensureLogInit() {
	dbg::LogState *state = dbg::getLogState();
	if (state->isInit) {
		return;
	}

	std::lock_guard<std::mutex> lock(state->mutex);
	if (!state->isInit) {
		initLogUnlocked(state, nullptr);
	}
}

static void writeLogPrefix(
	FILE *file,
	dbg::LogLevel level,
	const dbg::LogConfig *config,
	const char *srcFile,
	int line) {
	if (config->shouldPrintDateTime) {
		std::time_t now = std::time(nullptr);
		std::tm localTime = {};
#if defined(_WIN32)
		localtime_s(&localTime, &now);
#else
		localtime_r(&now, &localTime);
#endif
		char timeText[32];
		std::strftime(timeText, sizeof(timeText), "%Y-%m-%d %H:%M:%S", &localTime);
		std::fprintf(file, "%s ", timeText);
	}

	std::fprintf(file, "[%s]", dbg::logLevelName(level));

	if (config->shouldPrintThread) {
		std::fprintf(file, " [%s]", dbg::getThreadName());
	}

	if (config->shouldPrintFileLine && srcFile != nullptr && line > 0) {
		std::fprintf(file, " %s:%d", srcFile, line);
	}

	std::fprintf(file, " ");
}

static void writeLogLine(
	FILE *file,
	dbg::LogLevel level,
	const dbg::LogConfig *config,
	const char *srcFile,
	int line,
	const char *text,
	bool shouldFlush) {
	writeLogPrefix(file, level, config, srcFile, line);
	std::fprintf(file, "%s\n", safeText(text, ""));

	if (shouldFlush) {
		std::fflush(file);
	}
}

static void destroyGpuTimerRecords(dbg::GpuTimerState *timer) {
	for (dbg::GpuTimerRecord &record : timer->records) {
		if (record.startEvent != nullptr) {
			cudaEventDestroy(record.startEvent);
			record.startEvent = nullptr;
		}

		if (record.stopEvent != nullptr) {
			cudaEventDestroy(record.stopEvent);
			record.stopEvent = nullptr;
		}
	}

	timer->records.clear();
	timer->stack.clear();
}

static int appendGpuTimerRecord(
	dbg::GpuTimerState *timer,
	const char *name,
	int parentIdx,
	int depth,
	cudaStream_t stream) {
	const int recordIdx = int(timer->records.size());

	timer->records.emplace_back();
	dbg::GpuTimerRecord *record = &timer->records.back();
	record->name = safeText(name, "");
	record->parentIdx = parentIdx;
	record->depth = depth;
	record->stream = stream;
	record->startEvent = nullptr;
	record->stopEvent = nullptr;
	record->startOffset_ms = 0.0f;
	record->elapsed_ms = 0.0f;
	record->childElapsed_ms = 0.0f;

	CUDA_CHECK(cudaEventCreate(&record->startEvent));
	CUDA_CHECK(cudaEventCreate(&record->stopEvent));
	CUDA_CHECK(cudaEventRecord(record->startEvent, stream));

	return recordIdx;
}

struct ThreadGpuTimerOwner {
	dbg::GpuTimerState state = {};

	~ThreadGpuTimerOwner() {
		if (state.isInit && !state.isActive) {
			dbg::destroyGpuTimer(&state);
		}
	}
};

//==============================================================================
// CUDA error checking
//==============================================================================

namespace dbg {

void cudaCheck(
	cudaError_t result,
	const char *expr,
	const char *file,
	int line) {
	if (result == cudaSuccess) {
		return;
	}

	std::fprintf(
		stderr,
		"CUDA error at %s:%d\n"
		"  expression: %s\n"
		"  error: %s (%s)\n",
		safeText(file, "<unknown>"),
		line,
		safeText(expr, "<unknown>"),
		cudaGetErrorName(result),
		cudaGetErrorString(result));
	std::fflush(stderr);
	std::abort();
}

void cudaCheckLast(const char *file, int line) {
	cudaCheck(cudaGetLastError(), "cudaGetLastError()", file, line);
}

void cudaCheckSync(const char *file, int line) {
	cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize()", file, line);
}

//==============================================================================
// Logging
//==============================================================================

LogConfig defaultLogConfig() {
	LogConfig config = {};
	config.logDir = ".";
	config.filePrefix = "dbg";
	config.shouldAppend = false;
	config.shouldFlush = true;
	config.shouldPrintDateTime = true;
	config.shouldPrintThread = true;
	config.shouldPrintFileLine = true;
	return config;
}

LogState *getLogState() {
	static LogState state = {};
	return &state;
}

void initLog(const LogConfig *config) {
	LogState *state = getLogState();
	std::lock_guard<std::mutex> lock(state->mutex);
	initLogUnlocked(state, config);
}

void shutdownLog() {
	LogState *state = getLogState();
	std::lock_guard<std::mutex> lock(state->mutex);

	if (state->file != nullptr) {
		std::fflush(state->file);
		std::fclose(state->file);
		state->file = nullptr;
	}

	state->isInit = false;
	state->filePath[0] = '\0';
}

void flushLog() {
	ensureLogInit();

	LogState *state = getLogState();
	std::lock_guard<std::mutex> lock(state->mutex);

	if (state->file != nullptr) {
		std::fflush(state->file);
	}

	std::fflush(stdout);
	std::fflush(stderr);
}

void setThreadName(const char *name) {
	char *threadName = threadNameBuffer();
	std::snprintf(threadName, THREAD_NAME_CAPACITY, "%s", safeText(name, "thread"));
}

const char *getThreadName() {
	char *threadName = threadNameBuffer();

	if (threadName[0] == '\0') {
		std::snprintf(threadName, THREAD_NAME_CAPACITY, "thread");
	}

	return threadName;
}

const char *logLevelName(LogLevel level) {
	switch (level) {
		case LogLevel::Info: return "info";
		case LogLevel::Warning: return "warning";
		case LogLevel::Error: return "error";
		case LogLevel::Debug: return "debug";
		case LogLevel::Timer: return "timer";
		default: return "unknown";
	}
}

void logString(
	LogLevel level,
	LogDest dest,
	const char *file,
	int line,
	const char *text) {
	ensureLogInit();

	LogState *state = getLogState();
	std::lock_guard<std::mutex> lock(state->mutex);

	const bool hasStdout = logDestHasStdout(dest);
	const bool hasStderr = logDestHasStderr(dest);
	const bool hasFile = logDestHasFile(dest);

	if (hasStdout) {
		writeLogLine(
			stdout,
			level,
			&state->config,
			file,
			line,
			text,
			state->config.shouldFlush);
	}

	if (hasStderr) {
		writeLogLine(
			stderr,
			level,
			&state->config,
			file,
			line,
			text,
			state->config.shouldFlush);
	}

	if (hasFile) {
		FILE *logFile = state->file != nullptr ? state->file : stderr;
		writeLogLine(
			logFile,
			level,
			&state->config,
			file,
			line,
			text,
			state->config.shouldFlush);
	}
}

void vlogf(
	LogLevel level,
	LogDest dest,
	const char *file,
	int line,
	const char *fmt,
	va_list args) {
	char stackText[4096];

	va_list sizeArgs;
	va_copy(sizeArgs, args);
	const int numChars = std::vsnprintf(
		stackText,
		sizeof(stackText),
		safeText(fmt, ""),
		sizeArgs);
	va_end(sizeArgs);

	if (numChars < 0) {
		logString(level, dest, file, line, "<log format error>");
		return;
	}

	if (size_t(numChars) < sizeof(stackText)) {
		logString(level, dest, file, line, stackText);
		return;
	}

	std::vector<char> text(size_t(numChars) + 1);
	va_list textArgs;
	va_copy(textArgs, args);
	std::vsnprintf(text.data(), text.size(), safeText(fmt, ""), textArgs);
	va_end(textArgs);

	logString(level, dest, file, line, text.data());
}

void logf(
	LogLevel level,
	LogDest dest,
	const char *file,
	int line,
	const char *fmt,
	...) {
	va_list args;
	va_start(args, fmt);
	vlogf(level, dest, file, line, fmt, args);
	va_end(args);
}

//==============================================================================
// CPU timers
//==============================================================================

CpuTimerState *getCpuTimerState() {
	static thread_local CpuTimerState state = {};
	return &state;
}

void reserveTimerRecords(int capacity) {
	CpuTimerState *timer = getCpuTimerState();
	DBG_ASSERT(!timer->isActive);

	if (timer->isActive || capacity <= 0) {
		return;
	}

	timer->records.reserve(size_t(capacity));
	timer->stack.reserve(size_t(capacity));
}

void clearTimerReport() {
	CpuTimerState *timer = getCpuTimerState();
	DBG_ASSERT(!timer->isActive);

	if (timer->isActive) {
		return;
	}

	timer->records.clear();
	timer->stack.clear();
	timer->hasReport = false;
}

const CpuTimerState *getLastTimerReport() {
	const CpuTimerState *timer = getCpuTimerState();
	return timer->hasReport ? timer : nullptr;
}

void beginTimer(const char *name) {
	CpuTimerState *timer = getCpuTimerState();
	DBG_ASSERT(!timer->isActive);

	if (timer->isActive) {
		return;
	}

	timer->records.clear();
	timer->stack.clear();
	timer->hasReport = false;
	timer->isActive = true;

	if (timer->records.capacity() == 0) {
		timer->records.reserve(DEV_TIMER_DEFAULT_RECORD_CAPACITY);
		timer->stack.reserve(DEV_TIMER_DEFAULT_RECORD_CAPACITY);
	}

	const auto now = std::chrono::steady_clock::now();
	timer->rootStartTime = now;

	CpuTimerRecord record = {};
	record.name = safeText(name, "root");
	record.parentIdx = INVALID_IDX;
	record.depth = 0;
	record.startOffset_ms = 0.0;
	record.elapsed_ms = 0.0;
	record.childElapsed_ms = 0.0;

	timer->records.push_back(record);

	CpuTimerFrame frame = {};
	frame.recordIdx = 0;
	frame.startTime = now;
	timer->stack.push_back(frame);
}

void endTimer() {
	CpuTimerState *timer = getCpuTimerState();
	DBG_ASSERT(timer->isActive);

	if (!timer->isActive) {
		return;
	}

	DBG_ASSERT(timer->stack.size() == 1);
	if (timer->stack.size() != 1) {
		return;
	}

	const auto now = std::chrono::steady_clock::now();
	const CpuTimerFrame frame = timer->stack.back();
	timer->stack.pop_back();

	CpuTimerRecord *record = &timer->records[size_t(frame.recordIdx)];
	record->elapsed_ms = elapsedMs(frame.startTime, now);
	record->startOffset_ms = 0.0;

	timer->isActive = false;
	timer->hasReport = true;
}

void beginSubTimer(const char *name) {
	CpuTimerState *timer = getCpuTimerState();
	DBG_ASSERT(timer->isActive && !timer->stack.empty());

	if (!timer->isActive || timer->stack.empty()) {
		return;
	}

	const auto now = std::chrono::steady_clock::now();
	const int parentIdx = timer->stack.back().recordIdx;
	const int recordIdx = int(timer->records.size());

	CpuTimerRecord record = {};
	record.name = safeText(name, "");
	record.parentIdx = parentIdx;
	record.depth = int(timer->stack.size());
	record.startOffset_ms = elapsedMs(timer->rootStartTime, now);
	record.elapsed_ms = 0.0;
	record.childElapsed_ms = 0.0;

	timer->records.push_back(record);

	CpuTimerFrame frame = {};
	frame.recordIdx = recordIdx;
	frame.startTime = now;
	timer->stack.push_back(frame);
}

void endSubTimer() {
	CpuTimerState *timer = getCpuTimerState();
	DBG_ASSERT(timer->isActive && timer->stack.size() > 1);

	if (!timer->isActive || timer->stack.size() <= 1) {
		return;
	}

	const auto now = std::chrono::steady_clock::now();
	const CpuTimerFrame frame = timer->stack.back();
	timer->stack.pop_back();

	CpuTimerRecord *record = &timer->records[size_t(frame.recordIdx)];
	record->elapsed_ms = elapsedMs(frame.startTime, now);

	if (record->parentIdx != INVALID_IDX) {
		timer->records[size_t(record->parentIdx)].childElapsed_ms += record->elapsed_ms;
	}
}

bool isTimerActive() {
	return getCpuTimerState()->isActive;
}

int getTimerDepth() {
	const CpuTimerState *timer = getCpuTimerState();

	if (!timer->isActive || timer->stack.empty()) {
		return 0;
	}

	return int(timer->stack.size()) - 1;
}

void printTimerReport(FILE *file, const CpuTimerState *timer) {
	if (file == nullptr) {
		return;
	}

	std::string report;
	appendCpuTimerReport(&report, timer);
	std::fwrite(report.data(), 1, report.size(), file);
}

void logTimerReport(const CpuTimerState *timer, LogDest dest) {
	std::string report;
	appendCpuTimerReport(&report, timer);
	logString(LogLevel::Timer, dest, nullptr, 0, report.c_str());
}

//==============================================================================
// GPU timers
//==============================================================================

GpuTimerState *getGpuTimerState() {
	static thread_local ThreadGpuTimerOwner owner = {};
	return &owner.state;
}

void initGpuTimer(GpuTimerState *timer, int reserveRecordCapacity) {
	DBG_ASSERT(timer != nullptr);

	if (timer == nullptr) {
		return;
	}

	DBG_ASSERT(!timer->isActive);
	if (timer->isActive) {
		return;
	}

	destroyGpuTimerRecords(timer);

	timer->isInit = true;
	timer->isActive = false;
	timer->hasReport = false;
	timer->device = INVALID_IDX;

	CUDA_CHECK(cudaGetDevice(&timer->device));

	if (reserveRecordCapacity > 0) {
		timer->records.reserve(size_t(reserveRecordCapacity));
		timer->stack.reserve(size_t(reserveRecordCapacity));
	}
}

void destroyGpuTimer(GpuTimerState *timer) {
	DBG_ASSERT(timer != nullptr);

	if (timer == nullptr) {
		return;
	}

	DBG_ASSERT(!timer->isActive);
	if (timer->isActive) {
		return;
	}

	destroyGpuTimerRecords(timer);
	timer->isInit = false;
	timer->isActive = false;
	timer->hasReport = false;
	timer->device = INVALID_IDX;
}

void reserveGpuTimerRecords(int capacity) {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(!timer->isActive);

	if (timer->isActive || capacity <= 0) {
		return;
	}

	timer->records.reserve(size_t(capacity));
	timer->stack.reserve(size_t(capacity));
}

void clearGpuTimerReport() {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(!timer->isActive);

	if (timer->isActive) {
		return;
	}

	destroyGpuTimerRecords(timer);
	timer->hasReport = false;
}

const GpuTimerState *getLastGpuTimerReport() {
	const GpuTimerState *timer = getGpuTimerState();
	return timer->hasReport ? timer : nullptr;
}

void beginGpuTimer(const char *name, cudaStream_t stream) {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(!timer->isActive);

	if (timer->isActive) {
		return;
	}

	if (!timer->isInit) {
		initGpuTimer(timer);
	}

	clearGpuTimerReport();

	timer->isActive = true;
	timer->hasReport = false;

	const int recordIdx = appendGpuTimerRecord(
		timer,
		safeText(name, "gpu"),
		INVALID_IDX,
		0,
		stream);
	timer->stack.push_back(recordIdx);
}

void endGpuTimer() {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(timer->isActive);

	if (!timer->isActive) {
		return;
	}

	DBG_ASSERT(timer->stack.size() == 1);
	if (timer->stack.size() != 1) {
		return;
	}

	const int rootIdx = timer->stack.back();
	timer->stack.pop_back();

	GpuTimerRecord *root = &timer->records[size_t(rootIdx)];
	CUDA_CHECK(cudaEventRecord(root->stopEvent, root->stream));
	CUDA_CHECK(cudaEventSynchronize(root->stopEvent));

	for (GpuTimerRecord &record : timer->records) {
		CUDA_CHECK(cudaEventElapsedTime(
			&record.elapsed_ms,
			record.startEvent,
			record.stopEvent));

		if (&record == root) {
			record.startOffset_ms = 0.0f;
		} else {
			CUDA_CHECK(cudaEventElapsedTime(
				&record.startOffset_ms,
				root->startEvent,
				record.startEvent));
		}

		record.childElapsed_ms = 0.0f;
	}

	for (const GpuTimerRecord &record : timer->records) {
		if (record.parentIdx != INVALID_IDX) {
			timer->records[size_t(record.parentIdx)].childElapsed_ms += record.elapsed_ms;
		}
	}

	timer->isActive = false;
	timer->hasReport = true;
}

void beginGpuSubTimer(const char *name, cudaStream_t stream) {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(timer->isActive && !timer->stack.empty());

	if (!timer->isActive || timer->stack.empty()) {
		return;
	}

	const int parentIdx = timer->stack.back();
	const GpuTimerRecord *parent = &timer->records[size_t(parentIdx)];

	DBG_ASSERT(stream == parent->stream);
	if (stream != parent->stream) {
		return;
	}

	const int recordIdx = appendGpuTimerRecord(
		timer,
		name,
		parentIdx,
		int(timer->stack.size()),
		stream);
	timer->stack.push_back(recordIdx);
}

void beginGpuSubTimer(const char *name) {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(timer->isActive && !timer->stack.empty());

	if (!timer->isActive || timer->stack.empty()) {
		return;
	}

	const int parentIdx = timer->stack.back();
	beginGpuSubTimer(name, timer->records[size_t(parentIdx)].stream);
}

void endGpuSubTimer() {
	GpuTimerState *timer = getGpuTimerState();
	DBG_ASSERT(timer->isActive && timer->stack.size() > 1);

	if (!timer->isActive || timer->stack.size() <= 1) {
		return;
	}

	const int recordIdx = timer->stack.back();
	timer->stack.pop_back();

	GpuTimerRecord *record = &timer->records[size_t(recordIdx)];
	CUDA_CHECK(cudaEventRecord(record->stopEvent, record->stream));
}

bool isGpuTimerActive() {
	return getGpuTimerState()->isActive;
}

int getGpuTimerDepth() {
	const GpuTimerState *timer = getGpuTimerState();

	if (!timer->isActive || timer->stack.empty()) {
		return 0;
	}

	return int(timer->stack.size()) - 1;
}

void printGpuTimerReport(FILE *file, const GpuTimerState *timer) {
	if (file == nullptr) {
		return;
	}

	std::string report;
	appendGpuTimerReport(&report, timer);
	std::fwrite(report.data(), 1, report.size(), file);
}

void logGpuTimerReport(const GpuTimerState *timer, LogDest dest) {
	std::string report;
	appendGpuTimerReport(&report, timer);
	logString(LogLevel::Timer, dest, nullptr, 0, report.c_str());
}

} // namespace dbg
