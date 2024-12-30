#pragma once

#include <cuda_runtime.h>

#include "basic_buffer.hpp"
#include "mathprim/core/cuda_utils.cuh"
#include "mathprim/core/dim.hpp"

namespace mathprim {
namespace backend::cuda {
namespace internal {

void *alloc(size_t size) {
  void *ptr;
  int ret = cudaMalloc(&ptr, size);
  if (ret != cudaSuccess) {
    throw std::bad_alloc{};
  }

#if MATHPRIM_VERBOSE_MALLOC
  printf("CUDA: Allocated %zu bytes at %p\n", size, ptr);
#endif
  return ptr;
}

void free(void *ptr) noexcept {
  assert_success(cudaFree(ptr));
#ifdef MATHPRIM_VERBOSE_MALLOC
  printf("CUDA: Freed %p\n", ptr);
#endif
}
} // namespace internal

template <typename T> basic_buffer<T> make_buffer(const dim_t &shape) {
  size_t byte_size = shape.numel() * sizeof(T);
  T *ptr = static_cast<T *>(internal::alloc(byte_size));
  // TODO: we do not support pitch for now
  dim_t stride = make_default_stride(shape);
  return {shape, stride, ptr, device_t::cuda, internal::free};
}

} // namespace backend::cuda

template <typename T> struct buffer_traits<T, device_t::cuda> {
  static constexpr size_t alloc_alignment = 128;

  static constexpr basic_buffer<T> make_buffer(const dim_t &shape) {
    return backend::cuda::make_buffer<T>(shape);
  }
};

} // namespace mathprim