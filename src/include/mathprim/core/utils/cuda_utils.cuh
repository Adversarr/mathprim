#pragma once
#ifndef MATHPRIM_ENABLE_CUDA
#error "This file should only be included when CUDA is enabled."
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept> // IWYU pragma: export

#include "mathprim/core/dim.hpp"

namespace mathprim {
namespace backend::cuda::internal {
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
} // namespace backend::cuda::internal

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(index_t dim) {
  return {static_cast<unsigned int>(dim), 1, 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<1> &dim) {
  return {static_cast<unsigned int>(dim.x_), 1, 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<2> &dim) {
  return {static_cast<unsigned int>(dim.x_),
          static_cast<unsigned int>(internal::to_valid_size(dim.y_)), 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<3> &dim) {
  return {static_cast<unsigned int>(dim.x_), 
          static_cast<unsigned int>(internal::to_valid_size(dim.y_)),
          static_cast<unsigned int>(internal::to_valid_size(dim.z_))};
}

// We do not provide a conversion for dim<4> because it is unsafe to ignore the
// 4th dimension.
// MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<4> &dim) {
//   return {static_cast<unsigned int>(dim.x_), static_cast<unsigned int>(dim.y_),
//           static_cast<unsigned int>(dim.z_)};
// }

} // namespace mathprim
