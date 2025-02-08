#pragma once
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>
namespace mp = mathprim;
using namespace mp::literal;

#define cts_begin(name, loop_count)                                            \
  cudaEvent_t name##_start, name##_stop;                                       \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&name##_start));                 \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&name##_stop));                  \
  for (int i = 0; i < (loop_count); i++) {                                     \
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(name##_start))

#define cts_end(name)                                                          \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(name##_stop));                   \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventSynchronize(name##_stop));              \
  float milliseconds = 0;                                                      \
  MATHPRIM_CUDA_CHECK_SUCCESS(                                                 \
      cudaEventElapsedTime(&milliseconds, name##_start, name##_stop));         \
  printf("(%s) Elapsed time: %f ms\n", #name, milliseconds);                   \
  }                                                                            \
  do {                                                                         \
  } while (0)