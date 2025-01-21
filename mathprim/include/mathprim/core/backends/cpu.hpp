#pragma once

#include <cstdlib>
#include <cstring>
#include <new>

#include "mathprim/core/buffer_view.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim {

namespace backend::cpu {

namespace internal {
void free(void* ptr) noexcept {
#if MATHPRIM_VERBOSE_MALLOC
  printf("CPU: Free %p\n", ptr);
#endif
  std::free(ptr);
}

// 128-byte alignment.
void* alloc(size_t size) {
#ifdef MATPRIM_BACKEND_CPU_NO_ALIGNMENT
  void* ptr = std::malloc(size);
#else
  void* ptr = std::aligned_alloc(MATHPRIM_BACKEND_CPU_ALIGNMENT, size);
#endif
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

template <> struct buffer_backend_traits<device_t::cpu> {
  static constexpr size_t alloc_alignment = MATHPRIM_BACKEND_CPU_ALIGNMENT;

  static void* alloc(size_t mem_in_bytes) {
    return backend::cpu::internal::alloc(mem_in_bytes);
  }

  static void free(void* ptr) noexcept { backend::cpu::internal::free(ptr); }

  static void memset(void* ptr, int value, size_t mem_in_bytes) noexcept {
    ::memset(ptr, value, mem_in_bytes);
  }

  static void memcpy_host_to_device(void* dst, const void* src,
                                    size_t mem_in_bytes) noexcept {
    ::memcpy(dst, src, mem_in_bytes);
  }

  static void memcpy_device_to_host(void* dst, const void* src,
                                    size_t mem_in_bytes) noexcept {
    ::memcpy(dst, src, mem_in_bytes);
  }

  static void memcpy_device_to_device(void* dst, const void* src,
                                      size_t mem_in_bytes) noexcept {
    ::memcpy(dst, src, mem_in_bytes);
  }

  template <typename T, index_t N>
  static void view_copy(
      basic_buffer_view<T, N, device_t::cpu> dst,
      basic_buffer_view<const T, N, device_t::cpu> src) noexcept {
    if (dst.is_contiguous() and src.is_contiguous()) {
      ::memcpy(dst.data(), src.data(), src.numel() * sizeof(T));
    } else {
      for (const dim<N>& idx : dst.shape()) {
        dst(idx) = src(idx);
      }
    }
  }
};

}  // namespace mathprim
