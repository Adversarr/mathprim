#pragma once
#include <cuda_device_runtime_api.h>
#ifndef MATHPRIM_ENABLE_CUDA
#  error "This file should be included only when cuda is enabled."
#endif
#include <cuda_runtime.h>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"

namespace mathprim {

class cuda_error final : public std::runtime_error {
public:
  explicit cuda_error(cudaError_t error) : std::runtime_error(cudaGetErrorString(error)) {}
  cuda_error(const cuda_error &) = default;
  cuda_error(cuda_error &&) noexcept = default;
};

namespace device {
class cuda : public basic_device<cuda> {
public:
  static constexpr size_t alloc_alignment = 128;

  void *malloc_impl(size_t size) const {
    void *ptr = nullptr;
    if (const auto status = cudaMalloc(&ptr, size); status != cudaSuccess) {
      throw cuda_error(status);
    }
    MATHPRIM_ASSERT(ptr != nullptr);
    return ptr;
  }

  void free_impl(void *ptr) const noexcept {
    MATHPRIM_INTERNAL_CUDA_CHECK_SUCCESS(cudaFree(ptr));
  }

  void memset_impl(void *ptr, int value, size_t size) const {
    if (const auto status = (cudaMemset(ptr, value, size)); status != cudaSuccess) {
      throw cuda_error(status);
    }
  }

  const char *name_impl() const noexcept {
    return "cuda";
  }
};

template <>
struct device_traits<cuda> {
  static constexpr size_t alloc_alignment = 128;
};

template <>
struct basic_memcpy<cpu, cuda> {
  void operator()(void *dst, const void *src, size_t size) const {
    if (const auto status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); status != cudaSuccess) {
      throw cuda_error(status);
    }
  }
};

}  // namespace device

}  // namespace mathprim
