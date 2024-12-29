#pragma once

#include <cuda_runtime.h>

#include "basic_buffer.hpp"
#include "mathprim/core/cuda_utils.cuh"
#include "mathprim/core/dim.hpp"

namespace mathprim::backend::cuda {

namespace internal {
void* alloc(size_t size) {
  void* ptr;
  int ret = cudaMalloc(&ptr, size);
  if (ret != cudaSuccess) {
    throw std::bad_alloc{};
  }
  return ptr;
}

void free(void* ptr) noexcept {
  assert_success(cudaFree(ptr));
}
}  // namespace internal

template <typename T>
basic_buffer<T> make_buffer(const dim_t& shape) {
  size_t byte_size = shape.numel() * sizeof(T);
  T* ptr = static_cast<T*>(internal::alloc(byte_size));
  // TODO: we do not support pitch for now
  dim_t stride = make_default_stride(shape);
  return {shape, stride, ptr, internal::free};
}

}  // namespace mathprim::backend::cuda
