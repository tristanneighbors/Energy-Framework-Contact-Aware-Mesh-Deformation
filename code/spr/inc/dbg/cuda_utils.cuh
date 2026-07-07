#pragma once

/**
 * @file cuda_utils.cuh
 * @brief CUDA error checks, launch checks, and small indexing helpers.
 *
 * `CUDA_CHECK(expr)` reports the expression, file, line, CUDA error name, and
 * CUDA error string before aborting. When CUDA checks are disabled, the checked
 * expression is still evaluated so allocation and launch-side effects remain.
 */

#include "dbg_config.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

//==============================================================================
// CUDA error checking
//==============================================================================

namespace dbg {

void cudaCheck(
	cudaError_t result,
	const char *expr,
	const char *file,
	int line);

void cudaCheckLast(
	const char *file,
	int line);

void cudaCheckSync(
	const char *file,
	int line);

} // namespace dbg

#if DEV_ENABLE_CUDA_CHECK

#define CUDA_CHECK(expr) \
	dbg::cudaCheck((expr), #expr, __FILE__, __LINE__)

#define CUDA_LAUNCH_CHECK() \
	dbg::cudaCheckLast(__FILE__, __LINE__)

#define CUDA_SYNC_CHECK() \
	dbg::cudaCheckSync(__FILE__, __LINE__)

#else

// CUDA_CHECK still evaluates expr. Otherwise CUDA_CHECK(cudaMalloc(...)) would
// silently stop allocating when CUDA checks are disabled.
#define CUDA_CHECK(expr) ((void)(expr))

#define CUDA_LAUNCH_CHECK() ((void)0)
#define CUDA_SYNC_CHECK() ((void)0)

#endif

//==============================================================================
// Integer helpers
//==============================================================================

namespace dbg {

__host__ __device__ inline int divUp(int a, int b) {
	return (a + b - 1)/b;
}

__host__ __device__ inline unsigned int divUp(unsigned int a, unsigned int b) {
	return (a + b - 1)/b;
}

__host__ __device__ inline size_t divUp(size_t a, size_t b) {
	return (a + b - 1)/b;
}

__host__ __device__ inline int roundUp(int a, int b) {
	return divUp(a, b)*b;
}

__host__ __device__ inline unsigned int roundUp(unsigned int a, unsigned int b) {
	return divUp(a, b)*b;
}

__host__ __device__ inline size_t roundUp(size_t a, size_t b) {
	return divUp(a, b)*b;
}

__host__ __device__ inline int idx2(int x, int y, int width) {
	return x + width*y;
}

__host__ __device__ inline int idx3(
	int x,
	int y,
	int z,
	int width,
	int height) {
	return x + width*(y + height*z);
}

__host__ __device__ inline size_t idx2Size(
	size_t x,
	size_t y,
	size_t width) {
	return x + width*y;
}

__host__ __device__ inline size_t idx3Size(
	size_t x,
	size_t y,
	size_t z,
	size_t width,
	size_t height) {
	return x + width*(y + height*z);
}

__host__ __device__ inline int numBlocks(int numItems, int threadsPerBlock) {
	return divUp(numItems, threadsPerBlock);
}

__host__ __device__ inline int numBlocks(int numItems) {
	return divUp(numItems, DEFAULT_THREADS_PER_BLOCK);
}

} // namespace dbg

//==============================================================================
// Device launch helpers
//==============================================================================

namespace dbg {

__device__ inline int globalThreadIdx1d() {
	return int(blockIdx.x*blockDim.x + threadIdx.x);
}

__device__ inline int globalThreadStride1d() {
	return int(blockDim.x*gridDim.x);
}

__device__ inline int2 globalThreadIdx2d() {
	int2 out;
	out.x = int(blockIdx.x*blockDim.x + threadIdx.x);
	out.y = int(blockIdx.y*blockDim.y + threadIdx.y);
	return out;
}

__device__ inline int2 globalThreadStride2d() {
	int2 out;
	out.x = int(blockDim.x*gridDim.x);
	out.y = int(blockDim.y*gridDim.y);
	return out;
}

__device__ inline int3 globalThreadIdx3d() {
	int3 out;
	out.x = int(blockIdx.x*blockDim.x + threadIdx.x);
	out.y = int(blockIdx.y*blockDim.y + threadIdx.y);
	out.z = int(blockIdx.z*blockDim.z + threadIdx.z);
	return out;
}

__device__ inline int3 globalThreadStride3d() {
	int3 out;
	out.x = int(blockDim.x*gridDim.x);
	out.y = int(blockDim.y*gridDim.y);
	out.z = int(blockDim.z*gridDim.z);
	return out;
}

} // namespace dbg
