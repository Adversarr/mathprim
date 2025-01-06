#pragma once

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

inline void free(void *ptr) noexcept {
#ifdef MATHPRIM_VERBOSE_MALLOC
  printf("CUDA: Free %p\n", ptr);
#endif
  assert_success(cudaFree(ptr));
}

template <typename T, index_t N, typename = std::enable_if_t<N <= 3>>
__global__ void view_copy_impl(
    basic_buffer_view<T, N, device_t::cuda> dst,
    basic_buffer_view<const T, N, device_t::cuda> src) {
  const dim<N> idx = from_cuda_dim<N>(dim3(blockIdx));
  dst(idx) = src(idx);  // no check is required.
}

template <typename T>
__global__ void view_copy_impl4(
    basic_buffer_view<T, 4, device_t::cuda> dst,
    basic_buffer_view<const T, 4, device_t::cuda> src) {
  const dim<3> idx = from_cuda_dim<3>(dim3(blockIdx));
  const index_t w = ::mathprim::internal::to_valid_size(dst.shape(3));
  for (index_t i = 0; i < w; ++i) {
    dim<4> idx4{idx.x_, idx.y_, idx.z_, i};
    dst(idx4) = src(idx4);  // no check is required.
  }
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

  template <typename T, index_t N>
  static void view_copy(
      basic_buffer_view<T, N, device_t::cuda> dst,
      basic_buffer_view<const T, N, device_t::cuda> src) noexcept {
    if constexpr (N == 4) {
      dim3 grid = to_cuda_dim(dst.shape().xyz());
      backend::cuda::internal::view_copy_impl4<<<grid, 1>>>(dst, src);
    } else {
      dim3 grid = to_cuda_dim(dst.shape());
      backend::cuda::internal::view_copy_impl<<<grid, 1>>>(dst, src);
    };
  }
};

}  // namespace mathprim
