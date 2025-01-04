#pragma once

#include <cuda_runtime.h>

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
} // namespace internal

} // namespace backend::cuda

template <typename T> struct buffer_backend_traits<T, device_t::cuda> {
  static constexpr size_t alloc_alignment = 128;

  static void *alloc(size_t size) {
    return backend::cuda::internal::alloc(size);
  }

  static void free(void *ptr) noexcept { backend::cuda::internal::free(ptr); }

  static void memset(void *ptr, int value, size_t size) noexcept {
    backend::cuda::internal::assert_success(cudaMemset(ptr, value, size));
  }
};

} // namespace mathprim
