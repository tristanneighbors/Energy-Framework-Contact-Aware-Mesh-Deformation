#pragma once

/**
 * @file dbg_config.hpp
 * @brief Build toggles and common definitions for the debug utility.
 *
 * The `DEV_ENABLE_*` macros control whether assertions, logging, timers, and
 * CUDA checks are active. Disabled logging/timer macros compile to no-ops;
 * disabled `CUDA_CHECK(expr)` still evaluates `expr` in `cuda_utils.cuh`.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>

//==============================================================================
// Build toggles
//==============================================================================

#ifndef DEV_ENABLE_ASSERT
#define DEV_ENABLE_ASSERT 1
#endif

#ifndef DEV_ENABLE_LOG
#define DEV_ENABLE_LOG 1
#endif

#ifndef DEV_ENABLE_CPU_TIMER
#define DEV_ENABLE_CPU_TIMER 1
#endif

#ifndef DEV_ENABLE_GPU_TIMER
#define DEV_ENABLE_GPU_TIMER 1
#endif

#ifndef DEV_ENABLE_CUDA_CHECK
#define DEV_ENABLE_CUDA_CHECK 1
#endif

#ifndef DEV_TIMER_DEFAULT_RECORD_CAPACITY
#define DEV_TIMER_DEFAULT_RECORD_CAPACITY 256
#endif

#ifndef DEV_LOG_MAX_PATH_SIZE
#define DEV_LOG_MAX_PATH_SIZE 4096
#endif

//==============================================================================
// Assertions
//==============================================================================

#if DEV_ENABLE_ASSERT
#define DBG_ASSERT(expr) assert(expr)
#else
#define DBG_ASSERT(expr) ((void)0)
#endif

//==============================================================================
// Common definitions
//==============================================================================

namespace dbg {

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

constexpr int INVALID_IDX = -1;
constexpr int CUDA_WARP_SIZE = 32;
constexpr int DEFAULT_THREADS_PER_BLOCK = 256;

} // namespace dbg
