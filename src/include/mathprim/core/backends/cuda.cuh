#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"

namespace mathprim {
namespace backend::cuda {
namespace internal {

inline void *alloc(size_t size) {
  void *ptr = nullptr;
  int ret = cudaMalloc(&ptr, size);
  if (ret != cudaSuccess) {
    throw std::bad_alloc{};
  }

#if MATHPRIM_VERBOSE_MALLOC
  printf("CUDA: Allocated %zu bytes at %p\n", size, ptr);
#endif
  return ptr;
}

inline void free(void *ptr) noexcept {
#ifdef MATHPRIM_VERBOSE_MALLOC
  printf("CUDA: Free %p\n", ptr);
#endif
  assert_success(cudaFree(ptr));
}
}  // namespace internal

}  // namespace backend::cuda

template <> struct buffer_backend_traits<device_t::cuda> {
  static constexpr size_t alloc_alignment = 128;

  static void *alloc(size_t size) {
    return backend::cuda::internal::alloc(size);
  }

  static void free(void *ptr) noexcept { backend::cuda::internal::free(ptr); }

  static void memset(void *ptr, int value, size_t size) noexcept {
    backend::cuda::internal::assert_success(cudaMemset(ptr, value, size));
  }

  static void memcpy_host_to_device(void *dst, const void *src, size_t size) {
    auto status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      const char *error = cudaGetErrorString(status);
      throw memcpy_error{error};
    }
  }

  static void memcpy_device_to_host(void *dst, const void *src, size_t size) {
    auto status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      const char *error = cudaGetErrorString(status);
      throw memcpy_error{error};
    }
  }

  static void memcpy_device_to_device(void *dst, const void *src, size_t size) {
    auto status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess) {
      const char *error = cudaGetErrorString(status);
      throw memcpy_error{error};
    }
  }
};

}  // namespace mathprim
