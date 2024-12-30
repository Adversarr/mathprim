#pragma once
#include <assert.h>

#include <cstdint>
#include <cstdio>  // IWYU pragma: export
#include <memory>

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
#define MATHPRIM_ASSERT(cond) assert(cond)

// Enable/Disable copy constructor/assignment operator.
#define MATHPRIM_PRIMCOPY(cls, option)         \
  MATHPRIM_PRIMFUNC cls(const cls &) = option; \
  MATHPRIM_PRIMFUNC cls &operator=(const cls &) = option

#define MATHPRIM_COPY(cls, option) \
  cls(const cls &) = option;       \
  cls &operator=(const cls &) = option
#define MATHPRIM_PRIMMOVE(cls, option)    \
  MATHPRIM_PRIMFUNC cls(cls &&) = option; \
  MATHPRIM_PRIMFUNC cls &operator=(cls &&) = option
#define MATHPRIM_MOVE(cls, option) \
  cls(cls &&) = option;            \
  cls &operator=(cls &&) = option

#define MATHPRIM_STRINGIFY(x) #x

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
#    define MATHPRIM_INTERNAL_FATAL() \
      __builtin_trap();               \
      MATHPRIM_UNREACHABLE()
#  elif defined(_MSC_VER)
#    define MATHPRIM_INTERNAL_FATAL() \
      __debugbreak();                 \
      MATHPRIM_UNREACHABLE()
#  else
#    define MATHPRIM_INTERNAL_FATAL() \
      ((void)0);                      \
      MATHPRIM_UNREACHABLE()
#  endif
#endif

namespace mathprim {

///////////////////////////////////////////////////////////////////////////////
/// Indexing and scalar types.
///////////////////////////////////////////////////////////////////////////////
using std::size_t;

// Although indexing should use unsigned data type, but most libraries use
// a signed data type for indexing.
#if MATHPRIM_USE_LONG_INDEX
using index_t = std::int64_t;  ///< Type for indexing with 64-bit indices.
#  define MATHPRIM_INDEX_MAX 0x7FFFFFFFFFFFFFFF
#else
using index_t = std::int32_t;  ///< Type for indexing with 32-bit indices.
#  define MATHPRIM_INDEX_MAX 0x7FFFFFFF
#endif

constexpr size_t to_size(index_t i) {
  return static_cast<size_t>(i);
}

using f32_t = float;   ///< Type for 32-bit floating point numbers.
using f64_t = double;  ///< Type for 64-bit floating point numbers.

///////////////////////////////////////////////////////////////////////////////
/// Constants.
///////////////////////////////////////////////////////////////////////////////

constexpr index_t max_supported_dim = 4;  ///< The maximum supported dimension.

// Indicates this dimension does not exist logically.
constexpr index_t no_dim = 0;

// Indicates this dimension does not change under some operation.
constexpr index_t keep_dim = -1;

// TODO: currently, we only support cpu and gpu backends.
enum class device_t {
  cpu,      ///< CPU.
  cuda,     ///< NVidia GPU.
  dynamic,  ///< Reserved for untyped buffer view
};

enum class parallel_t {
  none,    ///< No parallelism.
  openmp,  ///< OpenMP. for cpu backend only
  cuda,    ///< CUDA.   for cuda backend only
};

///////////////////////////////////////////////////////////////////////////////
/// Declarations.
///////////////////////////////////////////////////////////////////////////////

template <index_t N>
struct dim;
using dim_t = dim<max_supported_dim>;  ///< The default dimensionality type for general buffers.

template <typename T>
class basic_buffer;
using f32_buffer = basic_buffer<f32_t>;
using f64_buffer = basic_buffer<f64_t>;
using index_buffer = basic_buffer<index_t>;
using float_buffer = f32_buffer;
using double_buffer = f64_buffer;

template <typename T, device_t dev>
struct buffer_backend_traits {
  // The alignment of the buffer.
  static constexpr size_t alloc_alignment = 0;
  // The default parallel backend.
  static constexpr parallel_t default_parallel = parallel_t::none;

  static void *alloc(size_t /* mem_in_bytes */);
  static void free(void * /* ptr */) noexcept;
  static void memset(void * /* ptr */, int /* value */, size_t /* mem_in_bytes */) noexcept;

  template <device_t from_dev>
  static void memcpy(void * /* dst */, const void * /* src */, size_t /* mem_in_bytes */) noexcept;
};

template <parallel_t par, device_t dev>
struct parallel_backend_traits {
  // currently, foreach_index is the only function to be implemented.
  template <typename Fn, typename... Args>
  static void foreach_index(const dim_t & /* grid_dim */, Fn && /* fn */, Args &&.../* args */) {
    static_assert(std::is_same_v<Fn, void>, "Unsupported parallel backend for given device");
  }
};

template <typename T>
using basic_buffer_ptr = std::unique_ptr<basic_buffer<T>>;

template <typename T, index_t N = max_supported_dim, device_t dev = device_t::dynamic>
class basic_buffer_view;

#define MATHPRIM_DECLARE_BUFFER_VIEW(tp, prefix)                                \
  template <index_t N = max_supported_dim, device_t dev = device_t::dynamic>    \
  using prefix##_buffer_view = basic_buffer_view<tp, N, dev>;                   \
  template <index_t N = max_supported_dim, device_t dev = device_t::dynamic>    \
  using const_##prefix##_buffer_view = basic_buffer_view<const tp, N, dev>;     \
  template <device_t dev = device_t::dynamic>                                   \
  using prefix##_buffer_view_1d = prefix##_buffer_view<1, dev>;                 \
  template <device_t dev = device_t::dynamic>                                   \
  using const_##prefix##_buffer_view_1d = const_##prefix##_buffer_view<1, dev>; \
  template <device_t dev = device_t::dynamic>                                   \
  using prefix##_buffer_view_2d = prefix##_buffer_view<2, dev>;                 \
  template <device_t dev = device_t::dynamic>                                   \
  using const_##prefix##_buffer_view_2d = const_##prefix##_buffer_view<2, dev>; \
  template <device_t dev = device_t::dynamic>                                   \
  using prefix##_buffer_view_3d = prefix##_buffer_view<3, dev>;                 \
  template <device_t dev = device_t::dynamic>                                   \
  using const_##prefix##_buffer_view_3d = const_##prefix##_buffer_view<3, dev>; \
  template <device_t dev = device_t::dynamic>                                   \
  using prefix##_buffer_view_4d = prefix##_buffer_view<4, dev>;                 \
  template <device_t dev = device_t::dynamic>                                   \
  using const_##prefix##_buffer_view_4d = const_##prefix##_buffer_view<4, dev>

MATHPRIM_DECLARE_BUFFER_VIEW(f32_t, f32);
MATHPRIM_DECLARE_BUFFER_VIEW(f64_t, f64);
MATHPRIM_DECLARE_BUFFER_VIEW(index_t, index);

}  // namespace mathprim
