#pragma once
#include <cuda_device_runtime_api.h>
#ifndef MATHPRIM_ENABLE_CUDA
#  error "This file should be included only when cuda is enabled."
#endif
#include <cuda_runtime.h>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"

namespace mathprim {

namespace device {
class cuda : public basic_device<cuda> {
public:
  static constexpr size_t alloc_alignment = 128;

  void *malloc_impl(size_t size) const {
    void *ptr = nullptr;
    auto status = cudaMalloc(&ptr, size);
    if (status != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
      throw std::bad_alloc{};
    }
    MATHPRIM_ASSERT(ptr != nullptr);
    return ptr;
  }

  void free_impl(void *ptr) const noexcept {
    MATHPRIM_INTERNAL_CUDA_CHECK_SUCCESS(cudaFree(ptr));
  }

  void memset_impl(void *ptr, int value, size_t size) const {
    MATHPRIM_INTERNAL_CUDA_CHECK_SUCCESS(cudaMemset(ptr, value, size));
  }

  const char *name_impl() const noexcept {
    return "cuda";
  }
};

template <>
struct device_traits<cuda> {
  static constexpr size_t alloc_alignment = 128;
};

}  // namespace device

}  // namespace mathprim
