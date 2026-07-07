#pragma once

/**
 * @file dbg.hpp
 * @brief Umbrella include for debug checks, logging, CUDA helpers, and timers.
 *
 * Runtime behavior is implemented in `src/dbg.cu`; users of logging or timers
 * should compile and link that translation unit.
 */

#include "dbg/dbg_config.hpp"
#include "dbg/cuda_utils.cuh"
#include "dbg/dbg_log.hpp"
#include "dbg/dbg_time.cuh"
