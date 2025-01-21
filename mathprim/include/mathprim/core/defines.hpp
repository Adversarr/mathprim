#pragma once
#include <assert.h>

#include <cstddef>
#include <cstdint>
#include <cstdio> // IWYU pragma: export
#include <memory>
#include <stdexcept>

///////////////////////////////////////////////////////////////////////////////
/// General Options
///////////////////////////////////////////////////////////////////////////////

// use 64-bit indices for indexing, this is not supported in most libraries.
#ifndef MATHPRIM_USE_LONG_INDEX
#define MATHPRIM_USE_LONG_INDEX 0
#endif

// verbose malloc/free
#ifndef MATHPRIM_VERBOSE_MALLOC
#define MATHPRIM_VERBOSE_MALLOC 0
#endif

#ifndef MATHPRIM_BACKEND_CPU_ALIGNMENT
#ifdef MATHPRIM_BACKEND_CPU_NO_ALIGNMENT
#define MATHPRIM_BACKEND_CPU_ALIGNMENT 0
#else
#define MATHPRIM_BACKEND_CPU_ALIGNMENT alignof(std::max_align_t)
#endif
#endif

#define MATHPRIM_CONCAT_IMPL(a, b) a##b
#define MATHPRIM_CONCAT(a, b) MATHPRIM_CONCAT_IMPL(a, b)
#ifndef MATHPRIM_CPU_BLAS
#ifdef MATHPRIM_ENABLE_BLAS
#define MATHPRIM_CPU_BLAS blas
#else
#define MATHPRIM_CPU_BLAS handmade
#endif
#endif
#define MATHPRIM_INTERNAL_CPU_BLAS_FALLBACK                                    \
  MATHPRIM_CONCAT(blas_impl_cpu_, MATHPRIM_CPU_BLAS)

///////////////////////////////////////////////////////////////////////////////
/// Feature detection
///////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER // for MSVC
#define MATHPRIM_FORCE_INLINE inline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define MATHPRIM_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined __CUDACC__
#define MATHPRIM_FORCE_INLINE __forceinline
#else
#define MATHPRIM_FORCE_INLINE inline
#endif

#ifdef __CUDACC__
#define MATHPRIM_HOST __host__
#define MATHPRIM_DEVICE __device__
#define MATHPRIM_GENERAL __host__ __device__
#else
#define MATHPRIM_HOST
#define MATHPRIM_DEVICE
#define MATHPRIM_GENERAL
#endif

///////////////////////////////////////////////////////////////////////////////
/// Shorthands for some common macros
///////////////////////////////////////////////////////////////////////////////
// For functions that are both inline and general (i.e., callable from both host
// and device code), we call them primary funcs (primfunc).
#define MATHPRIM_PRIMFUNC MATHPRIM_FORCE_INLINE MATHPRIM_GENERAL
#ifdef NDEBUG
#define MATHPRIM_ASSERT(cond) ((void)0)
#else
#define MATHPRIM_ASSERT(cond) assert(cond)
#endif

// Enable/Disable copy constructor/assignment operator.
#define MATHPRIM_PRIMCOPY(cls, option)                                         \
  MATHPRIM_PRIMFUNC cls(const cls &) = option;                                 \
  MATHPRIM_PRIMFUNC cls &operator=(const cls &) = option

#define MATHPRIM_COPY(cls, option)                                             \
  cls(const cls &) = option;                                                   \
  cls &operator=(const cls &) = option
#define MATHPRIM_PRIMMOVE(cls, option)                                         \
  MATHPRIM_PRIMFUNC cls(cls &&) = option;                                      \
  MATHPRIM_PRIMFUNC cls &operator=(cls &&) = option
#define MATHPRIM_MOVE(cls, option)                                             \
  cls(cls &&) = option;                                                        \
  cls &operator=(cls &&) = option

#ifndef MATHPRIM_UNREACHABLE
#if defined(__GNUC__) || defined(__clang__)
#define MATHPRIM_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define MATHPRIM_UNREACHABLE() __assume(0)
#else
#define MATHPRIM_UNREACHABLE() ((void)0)
#endif
#endif

#ifndef MATHPRIM_INTERNAL_FATAL
#if defined(__GNUC__) || defined(__clang__)
#define MATHPRIM_INTERNAL_FATAL(msg)                                           \
  fprintf(stderr, "%s\n", msg);                                                \
  __builtin_trap();                                                            \
  MATHPRIM_UNREACHABLE()
#elif defined(_MSC_VER)
#define MATHPRIM_INTERNAL_FATAL(msg)                                           \
  fprintf(stderr, "%s\n", msg);                                                \
  __debugbreak();                                                              \
  MATHPRIM_UNREACHABLE()
#else
#define MATHPRIM_INTERNAL_FATAL(msg)                                           \
  fprintf(stderr, "%s\n", msg);                                                \
  ((void)0);                                                                   \
  MATHPRIM_UNREACHABLE()
#endif
#endif

#ifndef MATHPRIM_PRAGMA_UNROLL
#ifdef __CUDA_ARCH__
#define MATHPRIM_PRAGMA_UNROLL _Pragma("unroll")
#elif defined(__clang__)
#define MATHPRIM_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define MATHPRIM_PRAGMA_UNROLL _Pragma("GCC unroll 4")
#elif defined(_MSC_VER)
#define MATHPRIM_PRAGMA_UNROLL __pragma(loop(ivdep))
#else
#define MATHPRIM_PRAGMA_UNROLL
#endif
#endif

namespace mathprim {

///////////////////////////////////////////////////////////////////////////////
/// Indexing and scalar types.
///////////////////////////////////////////////////////////////////////////////
using std::size_t; ///< unsigned integer type for size.

// Although indexing should use unsigned data type, but most libraries use
// a signed data type for indexing.
#if MATHPRIM_USE_LONG_INDEX
using index_t = std::int64_t; ///< Type for indexing with 64-bit indices.
#define MATHPRIM_INDEX_MAX 0x7FFFFFFFFFFFFFFF
#else
using index_t = std::int32_t; ///< Type for indexing with 32-bit indices.
#define MATHPRIM_INDEX_MAX 0x7FFFFFFF
#endif

constexpr size_t to_size(index_t i) { return static_cast<size_t>(i); }

using f32_t = float;  ///< Type for 32-bit floating point numbers.
using f64_t = double; ///< Type for 64-bit floating point numbers.

template <typename Flt> struct complex {
  Flt real_;
  Flt imag_;
};

using c32_t = complex<f32_t>; ///< Type for 32-bit complex numbers.
using c64_t = complex<f64_t>; ///< Type for 64-bit complex numbers.

///////////////////////////////////////////////////////////////////////////////
/// Constants.
///////////////////////////////////////////////////////////////////////////////

constexpr index_t max_ndim = 4; ///< The maximum supported dimension.

// Indicates this dimension does not exist logically. must be zero.
constexpr index_t no_dim = 0;

// Indicates this dimension does not change under some operation.
constexpr index_t keep_dim = -1;

// TODO: currently, we only support cpu and gpu backends.
enum class device_t {
  cpu,     ///< CPU.
  cuda,    ///< NVidia GPU.
  dynamic, ///< Reserved for untyped buffer view
};

// enum class par {
//   seq,     ///< No parallelism.
//   std,     ///< Standard C++ parallelism.
//   openmp,  ///< OpenMP. for cpu backend only
//   cuda,    ///< CUDA.   for cuda backend only
// };

namespace par {
class seq;
class std;
class openmp;
class cuda;
} // namespace par

///////////////////////////////////////////////////////////////////////////////
/// Errors
///////////////////////////////////////////////////////////////////////////////
// NOLINTBEGIN
#define MATHPRIM_INTERNAL_DECLARE_ERROR(name, derived)                         \
  class name : public std::derived {                                           \
  public:                                                                      \
    using std::derived::derived;                                               \
  }
// NOLINTEND

MATHPRIM_INTERNAL_DECLARE_ERROR(memcpy_error, runtime_error);
MATHPRIM_INTERNAL_DECLARE_ERROR(shape_error, runtime_error);
#undef MATHPRIM_INTERNAL_DECLARE_ERROR

#define MATHPRIM_INTERNAL_CUDA_ASSERT(cond, msg)                               \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("%s:%d::CUDA assertion(" #cond ") failed: %s\n", __FILE__,        \
             __LINE__, msg);                                                   \
    }                                                                          \
  } while (0)

///////////////////////////////////////////////////////////////////////////////
/// Forward Declarations.
///////////////////////////////////////////////////////////////////////////////

// TODO: should we use a simpler approach for blas

/// @brief BLAS backend
namespace blas {
template <typename T> struct blas_impl_cpu_handmade;
template <typename T> struct blas_impl_cpu_blas;
template <typename T> struct blas_impl_cpu_eigen;
} // namespace blas

/**
 * @brief Dimensionality type for general buffers.
 */
template <index_t N> struct dim;
using dim_t = dim<max_ndim>;

/**
 * @brief general buffer type.
 *
 * @tparam T the data type, need to be plain-old-data.
 */
template <typename T, index_t N, device_t dev> class basic_buffer;

/// @brief The buffer backend traits.
template <device_t dev> struct buffer_backend_traits;

/// @brief view of a buffer.
template <typename T, index_t N, device_t dev> class basic_buffer_view;

/// @brief iterator for buffer view.
template <typename T, index_t N, device_t dev> class basic_buffer_view_iterator;

///////////////////////////////////////////////////////////////////////////////
/// Parallelism
///////////////////////////////////////////////////////////////////////////////
/// @brief Parallel backend implementation.

/// @brief Parallel for loop
template <class par_impl> struct parfor;

///////////////////////////////////////////////////////////////////////////////
/// BLAS
///////////////////////////////////////////////////////////////////////////////
/// @brief BLAS backend selection fallback, do not use this to select backend.
template <typename T, device_t dev> struct blas_select_fallback;

template <typename T> struct blas_select_fallback<T, device_t::cpu> {
  using type = blas::MATHPRIM_INTERNAL_CPU_BLAS_FALLBACK<T>;
};

template <typename T> struct blas_select_fallback<T, device_t::cuda> {
  using type = blas::blas_impl_cpu_handmade<T>;
};

/// @brief BLAS backend selection (for user selection)
template <typename T, device_t dev> struct blas_select {
  using type = typename blas_select_fallback<T, dev>::type;
};

/// @brief BLAS backend selection (shortcut)
template <typename T, device_t dev>
using blas_select_t = typename blas_select<T, dev>::type;

///////////////////////////////////////////////////////////////////////////////
/// Aliases.
///////////////////////////////////////////////////////////////////////////////
using f32_buffer = basic_buffer<f32_t, max_ndim, device_t::dynamic>;
using f64_buffer = basic_buffer<f64_t, max_ndim, device_t::dynamic>;
using index_buffer = basic_buffer<index_t, max_ndim, device_t::dynamic>;
using float_buffer = f32_buffer;
using double_buffer = f64_buffer;

/// @brief default pointer to a buffer.
template <typename T, index_t N, device_t dev>
using basic_buffer_ptr = std::unique_ptr<basic_buffer<T, N, dev>>;

#define MATHPRIM_DECLARE_BUFFER_VIEW(tp, prefix)                               \
  template <index_t N = max_ndim, device_t dev = device_t::dynamic>            \
  using prefix##_buffer_view = basic_buffer_view<tp, N, dev>;                  \
  template <index_t N = max_ndim, device_t dev = device_t::dynamic>            \
  using const_##prefix##_buffer_view = basic_buffer_view<const tp, N, dev>;    \
  template <device_t dev = device_t::dynamic>                                  \
  using prefix##_buffer_view_1d = prefix##_buffer_view<1, dev>;                \
  template <device_t dev = device_t::dynamic>                                  \
  using const_##prefix##_buffer_view_1d =                                      \
      const_##prefix##_buffer_view<1, dev>;                                    \
  template <device_t dev = device_t::dynamic>                                  \
  using prefix##_buffer_view_2d = prefix##_buffer_view<2, dev>;                \
  template <device_t dev = device_t::dynamic>                                  \
  using const_##prefix##_buffer_view_2d =                                      \
      const_##prefix##_buffer_view<2, dev>;                                    \
  template <device_t dev = device_t::dynamic>                                  \
  using prefix##_buffer_view_3d = prefix##_buffer_view<3, dev>;                \
  template <device_t dev = device_t::dynamic>                                  \
  using const_##prefix##_buffer_view_3d =                                      \
      const_##prefix##_buffer_view<3, dev>;                                    \
  template <device_t dev = device_t::dynamic>                                  \
  using prefix##_buffer_view_4d = prefix##_buffer_view<4, dev>;                \
  template <device_t dev = device_t::dynamic>                                  \
  using const_##prefix##_buffer_view_4d = const_##prefix##_buffer_view<4, dev>

MATHPRIM_DECLARE_BUFFER_VIEW(f32_t, f32);
MATHPRIM_DECLARE_BUFFER_VIEW(f64_t, f64);
MATHPRIM_DECLARE_BUFFER_VIEW(index_t, index);

#undef MATHPRIM_DECLARE_BUFFER_VIEW

} // namespace mathprim
