#pragma once
#ifndef MATHPRIM_ENABLE_CUDA
#  error "This file should be included only when cuda is enabled."
#endif

#include <cuda_runtime.h>

#include <stdexcept>
#include <type_traits>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
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

// TODO: support pitching.

inline void free(void *ptr) noexcept {
#if MATHPRIM_VERBOSE_MALLOC
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

  static void free(void *ptr) noexcept {
    backend::cuda::internal::free(ptr);
  }

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

  template <typename T, index_t N>
  static void view_copy(basic_view<T, N, device_t::cuda> dst, basic_view<const T, N, device_t::cuda> src) noexcept {
    // TODO: blockDim=1 is not efficient for large N.
    if (!dst.contiguous() || !src.contiguous()) {
      throw std::invalid_argument{"view_copy: dst and src must be contiguous"};
    }

    MATHPRIM_ASSERT(dst.size() == src.size());
    cudaMemcpy(dst.data(), src.data(), dst.size() * sizeof(T));
  }
};

}  // namespace mathprim
