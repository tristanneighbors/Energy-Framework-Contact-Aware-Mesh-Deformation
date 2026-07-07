#pragma once

/**
 * @file dbg_log.hpp
 * @brief Thread-safe debug logging with optional file output.
 *
 * Logging initializes lazily with `defaultLogConfig()` unless `initLog()` is
 * called first. Writes are mutex-protected and can include timestamp, thread
 * name, and file/line prefixes according to `LogConfig`.
 */

#include "dbg_config.hpp"

#include <cstdarg>
#include <cstdio>
#include <mutex>

//==============================================================================
// Logging
//==============================================================================

namespace dbg {

enum class LogLevel {
	Info,
	Warning,
	Error,
	Debug,
	Timer,
};

enum class LogDest {
	Stdout,
	Stderr,
	File,
	StdoutAndFile,
	StderrAndFile,
};

/** Runtime logging options used by `initLog()`. */
struct LogConfig {
	const char *logDir = nullptr;
	const char *filePrefix = nullptr;

	bool shouldAppend = false;
	bool shouldFlush = false;
	bool shouldPrintDateTime = false;
	bool shouldPrintThread = false;
	bool shouldPrintFileLine = false;
};

/** Mutable logging backend state owned by the dbg runtime. */
struct LogState {
	LogConfig config = {};

	FILE *file = nullptr;
	bool isInit = false;

	std::mutex mutex;

	char filePath[DEV_LOG_MAX_PATH_SIZE] = {};
};

LogConfig defaultLogConfig();

LogState *getLogState();

void initLog(const LogConfig *config = nullptr);
void shutdownLog();
void flushLog();

void setThreadName(const char *name);
const char *getThreadName();

const char *logLevelName(LogLevel level);

void logString(
	LogLevel level,
	LogDest dest,
	const char *file,
	int line,
	const char *text);

void vlogf(
	LogLevel level,
	LogDest dest,
	const char *file,
	int line,
	const char *fmt,
	va_list args);

void logf(
	LogLevel level,
	LogDest dest,
	const char *file,
	int line,
	const char *fmt,
	...);

} // namespace dbg

//==============================================================================
// Logging macros
//==============================================================================

#if DEV_ENABLE_LOG

#define DBG_LOGF(...) \
	dbg::logf(dbg::LogLevel::Info, dbg::LogDest::File, __FILE__, __LINE__, __VA_ARGS__)

#define DBG_WARNF(...) \
	dbg::logf(dbg::LogLevel::Warning, dbg::LogDest::StderrAndFile, __FILE__, __LINE__, __VA_ARGS__)

#define DBG_ERRORF(...) \
	dbg::logf(dbg::LogLevel::Error, dbg::LogDest::StderrAndFile, __FILE__, __LINE__, __VA_ARGS__)

#define DBG_DEBUGF(...) \
	dbg::logf(dbg::LogLevel::Debug, dbg::LogDest::File, __FILE__, __LINE__, __VA_ARGS__)

#define DBG_TIMERF(...) \
	dbg::logf(dbg::LogLevel::Timer, dbg::LogDest::File, __FILE__, __LINE__, __VA_ARGS__)

#else

#define DBG_LOGF(...) ((void)0)
#define DBG_WARNF(...) ((void)0)
#define DBG_ERRORF(...) ((void)0)
#define DBG_DEBUGF(...) ((void)0)
#define DBG_TIMERF(...) ((void)0)

#endif
