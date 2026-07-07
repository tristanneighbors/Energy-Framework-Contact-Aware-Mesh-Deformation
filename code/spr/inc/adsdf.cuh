#pragma once

/**
 * @file adsdf.cuh
 * @brief Umbrella include for algebraic signed distance field construction, query, texture upload, and rendering.
 *
 * ADSDF stores one low-degree polynomial at each node of a regular 3D lattice.
 * Coefficients are kept in a global normalized power basis, so filtering first
 * interpolates coefficients and then evaluates one polynomial at the query
 * point. Construction currently fits local least-squares polynomials around
 * each node, translates them into the global basis, and stores packed `vec4`
 * coefficient groups for either linear-memory or CUDA texture queries.
 */

#include "adsdf/adsdf_types.cuh"
#include "adsdf/adsdf_poly.cuh"
#include "adsdf/sdf_sources.cuh"
#include "adsdf/adsdf_query.cuh"
#include "adsdf/adsdf_sample.cuh"
#include "adsdf/adsdf_build.cuh"
#include "adsdf/adsdf_texture.cuh"
#include "adsdf/adsdf_render.cuh"
