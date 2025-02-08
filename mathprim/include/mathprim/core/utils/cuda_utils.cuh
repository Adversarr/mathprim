#pragma once
#ifndef MATHPRIM_ENABLE_CUDA
#  error "This file should only be included when CUDA is enabled."
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>  // IWYU pragma: export

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

#define MATHPRIM_CUDA_CHECK_SUCCESS(expr)                                                                 \
  do {                                                                                                             \
    cudaError_t err = (expr);                                                                                      \
    if (err != cudaSuccess) {                                                                                      \
      fprintf(stderr, "CUDA check failed at %s:%d(%s): %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
      std::terminate();                                                                                            \
    }                                                                                                              \
  } while (false)

}  // namespace mathprim
