#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept> // IWYU pragma: export

#include "dim.hpp"

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
  return {static_cast<unsigned int>(dim.x), 1, 1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<2> &dim) {
  return {static_cast<unsigned int>(dim.x), static_cast<unsigned int>(dim.y),
          1};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<3> &dim) {
  return {static_cast<unsigned int>(dim.x), static_cast<unsigned int>(dim.y),
          static_cast<unsigned int>(dim.z)};
}

MATHPRIM_PRIMFUNC dim3 to_cuda_dim(const dim<4> &dim) {
  MATHPRIM_ASSERT(dim.w == no_dim && "CUDA does not support 4D indexing.");
  return {static_cast<unsigned int>(dim.x), static_cast<unsigned int>(dim.y),
          static_cast<unsigned int>(dim.z)};
}

} // namespace mathprim
