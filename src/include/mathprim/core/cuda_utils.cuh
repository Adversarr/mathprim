#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace mathprim::backend::cuda::internal {

template <typename Exception>
void check_success(cudaError_t status) {
  if (status != cudaSuccess) {
    throw Exception{cudaGetErrorString(status)};
  }
}

inline void assert_success(cudaError_t status) noexcept {
  if (status != cudaSuccess) {
    fprintf(stderr, "Terminate due to CUDA error: %s\n", cudaGetErrorString(status));
    std::terminate();
  }
}

}  // namespace mathprim::backend::cuda::internal
