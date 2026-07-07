#pragma once

/**
 * @file geometry.cuh
 * @brief Mesh, OBJ loading, triangle queries, and GPU BVH utilities.
 *
 * This module is intentionally independent of ADSDF code. ADSDF construction can
 * use the mesh/BVH distance-query surface here, but mesh geometry remains a
 * standalone utility for rendering and geometric queries.
 */

#include "geometry/geometry_types.cuh"
#include "geometry/geometry_query.cuh"
#include "geometry/geometry_mesh.cuh"
#include "geometry/geometry_topology.cuh"
#include "geometry/geometry_bvh.cuh"
#include "geometry/geometry_sdf.cuh"
#include "geometry/geometry_render.cuh"
#include "geometry/geometry_obj.cuh"
