#pragma once
#include <assert.h>

#include <cstdint>
#include <cstdio>  // IWYU pragma: export

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

using float32_t = float;   ///< Type for 32-bit floating point numbers.
using float64_t = double;  ///< Type for 64-bit floating point numbers.

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
  cpu,   ///< CPU.
  cuda,  ///< Nvidia GPU.
};

///////////////////////////////////////////////////////////////////////////////
/// Declarations.
///////////////////////////////////////////////////////////////////////////////

template <index_t N>
struct dim;

template <typename T>
class basic_buffer;

}  // namespace mathprim
