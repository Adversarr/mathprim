#pragma once
#ifndef MATHPRIM_ENABLE_CUDA
#  error "This file should only be included when CUDA is enabled."
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>  // IWYU pragma: export

#include "mathprim/core/dim.hpp"

#ifndef MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_1D
#  define MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_1D 64
#endif

#ifndef MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_2D
#  define MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_2D 8
#endif

#ifndef MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_3D
#  define MATHPRIM_CUDA_DEFAULT_BLOCK_SIZE_3D 4
#endif

namespace mathprim {

namespace cuda::internal {
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
}  // namespace cuda::internal

#define MATHPRIM_INTERNAL_CUDA_CHECK_SUCCESS(expr)                                                                 \
  do {                                                                                                             \
    cudaError_t err = (expr);                                                                                      \
    if (err != cudaSuccess) {                                                                                      \
      fprintf(stderr, "CUDA check failed at %s:%d(%s): %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
      std::terminate();                                                                                            \
    }                                                                                                              \
  } while (false)

}  // namespace mathprim
