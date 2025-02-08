#pragma once
#include <assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>  // IWYU pragma: export
#include <cstring>
#include <memory>  // IWYU pragma: export
#include <stdexcept>
#include <type_traits>  // IWYU pragma: export

///////////////////////////////////////////////////////////////////////////////
/// General Options
///////////////////////////////////////////////////////////////////////////////

// use 64-bit indices for indexing, this is not supported in most libraries.
#ifndef MATHPRIM_USE_LONG_INDEX
#  define MATHPRIM_USE_LONG_INDEX 0
#endif

// verbose malloc/free
#ifndef MATHPRIM_VERBOSE_MALLOC
#  define MATHPRIM_VERBOSE_MALLOC 0
#endif

#ifndef MATHPRIM_BACKEND_CPU_ALIGNMENT
#  ifdef MATHPRIM_BACKEND_CPU_NO_ALIGNMENT
#    define MATHPRIM_BACKEND_CPU_ALIGNMENT 0
#  else
#    define MATHPRIM_BACKEND_CPU_ALIGNMENT alignof(std::max_align_t)
#  endif
#endif

#define MATHPRIM_CONCAT_IMPL(a, b) a##b
#define MATHPRIM_CONCAT(a, b) MATHPRIM_CONCAT_IMPL(a, b)

///////////////////////////////////////////////////////////////////////////////
/// Feature detection
///////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER  // for MSVC
#  define MATHPRIM_FORCE_INLINE inline __forceinline
#elif defined __GNUC__  // for gcc on Linux/Apple OS X
#  define MATHPRIM_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined __CUDACC__
#  define MATHPRIM_FORCE_INLINE __forceinline
#else
#  define MATHPRIM_FORCE_INLINE inline
#endif

#ifdef __CUDACC__
#  define MATHPRIM_HOST __host__
#  define MATHPRIM_DEVICE __device__
#  define MATHPRIM_GENERAL __host__ __device__
#else
#  define MATHPRIM_HOST
#  define MATHPRIM_DEVICE
#  define MATHPRIM_GENERAL
#endif

///////////////////////////////////////////////////////////////////////////////
/// Shorthands for some common macros
///////////////////////////////////////////////////////////////////////////////
// For functions that are both inline and general (i.e., callable from both host
// and device code), we call them primary funcs (primfunc).
#define MATHPRIM_PRIMFUNC MATHPRIM_FORCE_INLINE MATHPRIM_GENERAL
#ifdef NDEBUG
#  define MATHPRIM_ASSERT(cond) ((void)0)
#  define MATHPRIM_CONSTEXPR
#else
#  define MATHPRIM_ASSERT(cond) assert(cond)
#  define MATHPRIM_CONSTEXPR constexpr
#endif

#define MATHPRIM_UNUSED(x) ((void)(x))

// Enable/Disable copy constructor/assignment operator.

#define MATHPRIM_INTERNAL_COPY(cls, option) \
  cls(const cls &) = option;                \
  cls &operator=(const cls &) = option

#define MATHPRIM_INTERNAL_MOVE(cls, option) \
  cls(cls &&) = option;                     \
  cls &operator=(cls &&) = option

#ifndef MATHPRIM_INTERNAL_MAYBE_UNUSED
#  define MATHPRIM_INTERNAL_MAYBE_UNUSED(x) (void)(x)
#endif

#ifndef MATHPRIM_UNREACHABLE
#  if defined(__GNUC__) || defined(__clang__)
#    define MATHPRIM_UNREACHABLE() __builtin_unreachable()
#  elif defined(_MSC_VER)
#    define MATHPRIM_UNREACHABLE() __assume(0)
#  else
#    define MATHPRIM_UNREACHABLE() ((void)0)
#  endif
#endif

#ifndef MATHPRIM_INTERNAL_FATAL
#  if defined(__GNUC__) || defined(__clang__)
#    define MATHPRIM_INTERNAL_FATAL(msg) \
      fprintf(stderr, "%s\n", msg);      \
      __builtin_trap();                  \
      MATHPRIM_UNREACHABLE()
#  elif defined(_MSC_VER)
#    define MATHPRIM_INTERNAL_FATAL(msg) \
      fprintf(stderr, "%s\n", msg);      \
      __debugbreak();                    \
      MATHPRIM_UNREACHABLE()
#  else
#    define MATHPRIM_INTERNAL_FATAL(msg) \
      fprintf(stderr, "%s\n", msg);      \
      ((void)0);                         \
      MATHPRIM_UNREACHABLE()
#  endif
#endif

#ifndef MATHPRIM_PRAGMA_UNROLL
#  ifdef __CUDA_ARCH__
#    define MATHPRIM_PRAGMA_UNROLL _Pragma("unroll")
#  elif defined(__clang__)
#    define MATHPRIM_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#  elif defined(__GNUC__)
#    define MATHPRIM_PRAGMA_UNROLL _Pragma("GCC unroll 4")
#  elif defined(_MSC_VER)
#    define MATHPRIM_PRAGMA_UNROLL __pragma(loop(ivdep))
#  else
#    define MATHPRIM_PRAGMA_UNROLL
#  endif
#endif

namespace mathprim {

///////////////////////////////////////////////////////////////////////////////
/// Indexing and scalar types.
///////////////////////////////////////////////////////////////////////////////
using std::size_t;  ///< unsigned integer type for size.

// Although indexing should use unsigned data type, but most libraries use
// a signed data type for indexing.
#if MATHPRIM_USE_LONG_INDEX
using index_t = std::int64_t;  ///< Type for indexing with 64-bit indices.
#  define MATHPRIM_INDEX_MAX 0x7FFFFFFFFFFFFFFF
#else
using index_t = std::int32_t;  ///< Type for indexing with 32-bit indices.
#  define MATHPRIM_INDEX_MAX 0x7FFFFFFF
#endif

using f32_t = float;   ///< Type for 32-bit floating point numbers.
using f64_t = double;  ///< Type for 64-bit floating point numbers.

template <typename Flt>
struct complex {
  Flt real_;
  Flt imag_;
};

using c32_t = complex<f32_t>;  ///< Type for 32-bit complex numbers.
using c64_t = complex<f64_t>;  ///< Type for 64-bit complex numbers.

///////////////////////////////////////////////////////////////////////////////
/// Constants.
///////////////////////////////////////////////////////////////////////////////

// Indicates this dimension does not change under some operation.
constexpr index_t keep_dim = -1;

///////////////////////////////////////////////////////////////////////////////
/// Device: CPU, CUDA, etc.
///////////////////////////////////////////////////////////////////////////////
namespace device {  // i.e. backends.

template <typename T>
struct device_traits;

template <typename Derived>
class basic_device {
public:
  /**
   * @brief Allocate memory on the device.
   *
   * @param size in bytes.
   * @return void* pointer to the allocated memory. guaranteed to be aligned.
   */
  void *malloc(size_t size) const {
    void *ptr = static_cast<const Derived *>(this)->malloc_impl(size);
    if (!ptr) {
      throw std::bad_alloc{};
    }
#if MATHPRIM_VERBOSE_MALLOC
    printf("%s: Allocated %zu bytes\n", name(), size);
#endif

#ifndef NDEBUG
    // Check alignment.
    const auto align = device_traits<Derived>::alloc_alignment;
    if (align > 0) {
      MATHPRIM_ASSERT(reinterpret_cast<uintptr_t>(ptr) % align == 0 && "Alignment error.");
    }
#endif
    return ptr;
  }

  /**
   * @brief Free memory on the device.
   *
   * @param ptr
   */
  void free(void *ptr) const {
    if (!ptr) {
      fprintf(stderr, "(WARN) %s: Freeing nullptr\n", name());
      return;
    }

#if MATHPRIM_VERBOSE_MALLOC
    printf("%s: Free %p\n", name(), ptr);
#endif
    static_cast<const Derived *>(this)->free_impl(ptr);
  }

  /**
   * @brief Set memory to a value.
   *
   * @param ptr Pointer to the memory, must not be nullptr.
   * @param value value to set.
   * @param size size in bytes.
   */
  void memset(void *ptr, int value, size_t size) const {
    if (!ptr) {
      throw std::invalid_argument{"Memsetting nullptr"};
    }

    if (size == 0) {
      return;
    }

    static_cast<const Derived *>(this)->memset_impl(ptr, value, size);
  }

  const char *name() const {
    return static_cast<const Derived *>(this)->name_impl();
  }
};

class cpu;

// This implementation is too essential to be here.
class cpu : public basic_device<cpu> {
public:
  void *malloc_impl(size_t size) const {
#if defined(_MSC_VER)
    void *ptr = _aligned_malloc(size, MATHPRIM_BACKEND_CPU_ALIGNMENT);
#else
    void *ptr = std::aligned_alloc(MATHPRIM_BACKEND_CPU_ALIGNMENT, size);
#endif
    if (!ptr) {
      throw std::bad_alloc{};
    }
    return ptr;
  }

  void free_impl(void *ptr) const noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
  }

  void memset_impl(void *ptr, int value, size_t size) const {
    std::memset(ptr, value, size);
  }

  const char *name_impl() const noexcept {
    return "cpu";
  }
};

// Include the <mathprim/core/devices/cuda.cuh> for the definition.
class cuda;

template <>
struct device_traits<cpu> {
  static constexpr size_t alloc_alignment = MATHPRIM_BACKEND_CPU_ALIGNMENT;
};

template <typename From, typename To>
struct basic_memcpy;

template <>
struct basic_memcpy<cpu, cpu> {
  void operator()(void *dst, const void *src, size_t size) const noexcept {
    std::memcpy(dst, src, size);
  }
};

}  // namespace device

///////////////////////////////////////////////////////////////////////////////
/// Errors
///////////////////////////////////////////////////////////////////////////////
#define MATHPRIM_INTERNAL_DECLARE_ERROR(name, derived) \
  class name final : public std::derived {             \
  public:                                              \
    using std::derived::derived;                       \
  }

MATHPRIM_INTERNAL_DECLARE_ERROR(memcpy_error, runtime_error);
MATHPRIM_INTERNAL_DECLARE_ERROR(shape_error, runtime_error);
#undef MATHPRIM_INTERNAL_DECLARE_ERROR

#define MATHPRIM_CUDA_ASSERT(cond, msg)                                                 \
  do {                                                                                  \
    if (!(cond)) {                                                                      \
      printf("%s:%d::CUDA assertion(" #cond ") failed: %s\n", __FILE__, __LINE__, msg); \
    }                                                                                   \
  } while (0)

///////////////////////////////////////////////////////////////////////////////
/// Forward Declarations.
///////////////////////////////////////////////////////////////////////////////
template <index_t N>
struct index_array;
template <index_t... svalues>
struct index_pack;
template <index_t... args>
struct index_seq {
  static constexpr index_t ndim = sizeof...(args);
};

template <index_t... svalues>
using shape_t = index_pack<svalues...>;
template <index_t... svalues>
using stride_t = index_pack<svalues...>;

/**
 * @brief general buffer type.
 *
 * @tparam T the data type, need to be plain-old-data.
 * @tparam sshape the shape of the buffer.
 * @tparam sstride the stride of the buffer.
 * @tparam dev the device type.
 */
template <typename T, typename sshape, typename sstride, typename dev>
class basic_buffer;

/**
 * @brief general view type.
 *
 * @tparam T the data type, need to be plain-old-data.
 * @tparam sshape the shape of the buffer.
 * @tparam sstride the stride of the buffer.
 * @tparam dev the device type.
 */
template <typename T, typename sshape, typename sstride, typename dev>
class basic_view;

/**
 * @brief iterator for view.
 *
 * @tparam T the data type, need to be plain-old-data.
 * @tparam sshape the shape of the buffer.
 * @tparam sstride the stride of the buffer.
 * @tparam dev the device type.
 */
template <typename T, typename sshape, typename sstride, typename dev>
struct dimension_iterator;

///////////////////////////////////////////////////////////////////////////////
/// Parallelism
///////////////////////////////////////////////////////////////////////////////
/// @brief Parallel backend implementation.

/// @brief Parallel for loop
namespace par {
template <class par_impl>
struct parfor;
}

///////////////////////////////////////////////////////////////////////////////
/// Aliases.
///////////////////////////////////////////////////////////////////////////////

#define MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(n, t, d)                                                   \
  using d##_##vec##n##t##_view_t = basic_view<t##_t, shape_t<n>, stride_t<sizeof(t##_t)>, device::d>; \
  using d##_##vec##n##t##_const_view_t = basic_view<const t##_t, shape_t<n>, stride_t<sizeof(t##_t)>, device::d>

MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(2, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(3, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(4, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(2, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(3, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(4, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(2, index, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(3, index, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(4, index, cpu);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(2, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(3, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(4, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(2, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(3, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(4, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(2, index, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(3, index, cuda);
MATHPRIM_INTERNAL_DECLARE_VEC_VIEW(4, index, cuda);
#undef MATHPRIM_INTERNAL_DECLARE_VEC_VIEW

using cpu_vecxf32_view_t = basic_view<f32_t, shape_t<-1>, stride_t<sizeof(f32_t)>, device::cpu>;
using cuda_vecxf32_view_t = basic_view<f32_t, shape_t<-1>, stride_t<sizeof(f32_t)>, device::cuda>;
using cpu_vecxf64_view_t = basic_view<f64_t, shape_t<-1>, stride_t<sizeof(f64_t)>, device::cpu>;
using cuda_vecxf64_view_t = basic_view<f64_t, shape_t<-1>, stride_t<sizeof(f64_t)>, device::cuda>;
using cpu_vecxindex_view_t = basic_view<index_t, shape_t<-1>, stride_t<sizeof(index_t)>, device::cpu>;
using cuda_vecxindex_view_t = basic_view<index_t, shape_t<-1>, stride_t<sizeof(index_t)>, device::cuda>;
using cpu_vecxf32_const_view_t = basic_view<const f32_t, shape_t<-1>, stride_t<sizeof(f32_t)>, device::cpu>;
using cuda_vecxf32_const_view_t = basic_view<const f32_t, shape_t<-1>, stride_t<sizeof(f32_t)>, device::cuda>;
using cpu_vecxf64_const_view_t = basic_view<const f64_t, shape_t<-1>, stride_t<sizeof(f64_t)>, device::cpu>;
using cuda_vecxf64_const_view_t = basic_view<const f64_t, shape_t<-1>, stride_t<sizeof(f64_t)>, device::cuda>;
using cpu_vecxindex_const_view_t = basic_view<const index_t, shape_t<-1>, stride_t<sizeof(index_t)>, device::cpu>;
using cuda_vecxindex_const_view_t = basic_view<const index_t, shape_t<-1>, stride_t<sizeof(index_t)>, device::cuda>;

#define MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(r, c, t, d)                                             \
  using d##_##mat##r##x##c##t##_view_t                                                             \
      = basic_view<t##_t, shape_t<r, c>, stride_t<(c) * sizeof(t##_t), sizeof(t##_t)>, device::d>; \
  using d##_##mat##r##x##c##t##_const_view_t                                                       \
      = basic_view<const t##_t, shape_t<r, c>, stride_t<(c) * sizeof(t##_t), sizeof(t##_t)>, device::d>
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 2, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 3, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 4, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 2, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 3, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 4, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 2, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 3, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 4, f32, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 2, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 3, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 4, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 2, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 3, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 4, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 2, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 3, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 4, f64, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 2, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 3, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 4, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 2, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 3, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 4, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 2, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 3, index, cpu);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 4, index, cpu);

MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 2, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 3, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 4, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 2, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 3, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 4, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 2, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 3, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 4, f32, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 2, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 3, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 4, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 2, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 3, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 4, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 2, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 3, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 4, f64, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 2, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 3, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(2, 4, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 2, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 3, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(3, 4, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 2, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 3, index, cuda);
MATHPRIM_INTERNAL_DECLARE_MAT_VIEW(4, 4, index, cuda);
#undef MATHPRIM_INTERNAL_DECLARE_MAT_VIEW

using cpu_matxxf32_view_t
    = basic_view<f32_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f32_t)>, device::cpu>;
using cpu_matxxf64_view_t
    = basic_view<f64_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f64_t)>, device::cpu>;
using cuda_matxxf32_view_t
    = basic_view<f32_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f32_t)>, device::cuda>;
using cuda_matxxf64_view_t
    = basic_view<f64_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f64_t)>, device::cuda>;
using cpu_matxxf32_const_view_t
    = basic_view<const f32_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f32_t)>, device::cpu>;
using cpu_matxxf64_const_view_t
    = basic_view<const f64_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f64_t)>, device::cpu>;
using cuda_matxxf32_const_view_t
    = basic_view<const f32_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f32_t)>, device::cuda>;
using cuda_matxxf64_const_view_t
    = basic_view<const f64_t, shape_t<keep_dim, keep_dim>, stride_t<keep_dim, sizeof(f64_t)>, device::cuda>;

///////////////////////////////////////////////////////////////////////////////

// Utility functions.
template <typename Integer, typename = std::enable_if_t<std::is_integral_v<Integer>>>
MATHPRIM_PRIMFUNC Integer up_div(Integer a, Integer b) noexcept {
  return (a + b - 1) / b;
}

}  // namespace mathprim