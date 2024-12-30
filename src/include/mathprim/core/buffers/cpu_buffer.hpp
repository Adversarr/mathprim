#pragma once

#include <cstdlib>

#include "basic_buffer.hpp"

namespace mathprim {

namespace backend::cpu {

namespace internal {
void free(void* ptr) noexcept {
  std::free(ptr);

#if MATHPRIM_VERBOSE_MALLOC
  printf("CPU: Freed %p\n", ptr);
#endif
}

// 128-byte alignment.
void* alloc_128(size_t size) {
  void* ptr = std::aligned_alloc(128, size);
  if (!ptr) {
    throw std::bad_alloc{};
  }

#if MATHPRIM_VERBOSE_MALLOC
  printf("CPU: Allocated %zu bytes at %p\n", size, ptr);
#endif

  return ptr;
}
}  // namespace internal

template <typename T>
basic_buffer<T> make_buffer(const dim_t& shape) {
  dim_t stride = make_default_stride(shape);
  size_t total = numel(shape) * sizeof(T);
  T* data = static_cast<T*>(internal::alloc_128(total));
  return {shape, stride, data, device_t::cpu, internal::free};
}

}  // namespace backend::cpu

template <typename T>
struct buffer_traits<T, device_t::cpu> {
  static constexpr size_t alloc_alignment = 128;

  static constexpr basic_buffer<T> make_buffer(const dim_t& shape) {
    return backend::cpu::make_buffer<T>(shape);
  }
};

}  // namespace mathprim
