#pragma once
#ifndef MATHPRIM_ENABLE_CUDA
#error "This file should only be included when CUDA is enabled."
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept> // IWYU pragma: export

#include "mathprim/core/dim.hpp"

#ifndef MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_1D
#define MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_1D 64
#endif

#ifndef MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_2D
#define MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_2D 8
#endif

#ifndef MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_3D
#define MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_3D 4
#endif

namespace mathprim {

namespace cuda::internal {
template <typename Exception> void check_success(cudaError_t status) {
  if (status != cudaSuccess) {
    throw Exception{cudaGetErrorString(status)};
  }
}

inline void assert_success(cudaError_t status) noexcept {
  if (status != cudaSuccess) {
    fprintf(stderr, "Terminate due to CUDA error: %s\n",
            cudaGetErrorString(status));
    std::terminate();
  }
}
} // namespace cuda::internal

template <index_t N> MATHPRIM_PRIMFUNC dim<N> from_cuda_dim(const dim3 &d);

template <> MATHPRIM_PRIMFUNC dim<1> from_cuda_dim(const dim3 &d) {
  return dim<1>{static_cast<index_t>(d.x)};
}

template <> MATHPRIM_PRIMFUNC dim<2> from_cuda_dim(const dim3 &d) {
  return dim<2>{static_cast<index_t>(d.x), static_cast<index_t>(d.y)};
}

template <> MATHPRIM_PRIMFUNC dim<3> from_cuda_dim(const dim3 &d) {
  return dim<3>{static_cast<index_t>(d.x), static_cast<index_t>(d.y),
                static_cast<index_t>(d.z)};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(index_t dim) {
  return {static_cast<unsigned int>(dim), 1, 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<1> &dim) {
  return {static_cast<unsigned int>(dim.x_), 1, 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<2> &dim) {
  return {static_cast<unsigned int>(dim.x_),
          static_cast<unsigned int>(internal::to_valid_index(dim.y_)), 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<3> &dim) {
  return {static_cast<unsigned int>(dim.x_),
          static_cast<unsigned int>(internal::to_valid_index(dim.y_)),
          static_cast<unsigned int>(internal::to_valid_index(dim.z_))};
}

} // namespace mathprim
