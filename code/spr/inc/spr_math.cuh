#pragma once

/**
 * @file spr_math.cuh
 * @brief Umbrella include for the CUDA-friendly small math types.
 *
 * This header collects fixed-size vectors and matrices, algebraic complex and
 * quaternion values, rotation groups, transform groups, random sampling helpers,
 * and small geometry utilities. The types are intended for small dimensions
 * that benefit from inline, header-only CUDA code.
 */

#include "spr_global_include.h"
#include "config.hpp"

#include "spr_math_detail/scalar_math.cuh"
#include "spr_math_detail/vec.cuh"
#include "spr_math_detail/mat.cuh"
#include "spr_math_detail/complex.cuh"
#include "spr_math_detail/quat.cuh"
#include "spr_math_detail/rot.cuh"
#include "spr_math_detail/sym.cuh"
#include "spr_math_detail/diag.cuh"
#include "spr_math_detail/rigid.cuh"
#include "spr_math_detail/sim.cuh"
#include "spr_math_detail/affine.cuh"
#include "spr_math_detail/random.cuh"
#include "spr_math_detail/geometry_util.cuh"
