#pragma once

#include <cstdlib>
#include <cstring>
#include <new>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"  // IWYU pragma: keep

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
void* alloc(size_t size) {
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

}  // namespace backend::cpu

template <typename T>
struct buffer_backend_traits<T, device_t::cpu> {
  static constexpr size_t alloc_alignment = 128;

  static void* alloc(size_t mem_in_bytes) { return backend::cpu::internal::alloc(mem_in_bytes); }

  static void free(void* ptr) noexcept { backend::cpu::internal::free(ptr); }

  static void memset(void* ptr, int value, size_t mem_in_bytes) noexcept {
    ::memset(ptr, value, mem_in_bytes);
  }
};

}  // namespace mathprim
