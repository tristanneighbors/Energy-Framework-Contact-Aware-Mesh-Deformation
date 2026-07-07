#pragma once

/**
 * @file sprcu_util.cuh
 * @brief Compatibility include for CUDA utility helpers.
 *
 * New code should include `dbg/cuda_utils.cuh` directly. This header remains so
 * older includes keep finding the CUDA check and indexing helpers.
 */

#include "dbg/cuda_utils.cuh"
