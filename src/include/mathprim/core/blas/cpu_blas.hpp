#pragma once

#include <cstddef>
#include <type_traits>

#include "/opt/homebrew/opt/openblas/include/cblas.h"
#include "basic_blas.hpp"
#include "helper_macros.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim {
namespace buffer_blas {

template <typename T>
struct backend_blas<T, blas_t::cpu_blas, device_t::cpu> {
  static constexpr device_t dev = device_t::cpu;
  using vector_view = basic_buffer_view<T, 1, dev>;
  using matrix_view = basic_buffer_view<T, 2, dev>;
  using const_vector_view = basic_buffer_view<const T, 1, dev>;
  using const_matrix_view = basic_buffer_view<const T, 2, dev>;

  // Level 1
  void copy(vector_view dst, const_vector_view src);
  void scal(T alpha, vector_view x);
  void swap(vector_view x, vector_view y);
  void axpy(T alpha, const_vector_view x, vector_view y);

  T dot(const_vector_view x, const_vector_view y);
  T norm(const_vector_view x);
  T asum(const_vector_view x);
  T amax(const_vector_view x);

  // Level 2
  // y <- alpha * A * x + beta * y
  void gemv(T alpha, const_matrix_view A, const_vector_view x, T beta,
            vector_view y);

  // Level 3
  // C <- alpha * A * B + beta * C
  void gemm(T alpha, const_matrix_view A, const_matrix_view B, T beta,
            matrix_view C);

  // element-wise operatons
  // y = x * y
  void emul(const_vector_view x, vector_view y);
  // y = x / y
  void ediv(const_vector_view x, vector_view y);
};

}  // namespace buffer_blas

// Set default fallback.
template <>
struct blas_select_fallback<device_t::cpu> {
  static constexpr blas_t value = blas_t::cpu_handmade;
};

}  // namespace mathprim

namespace mathprim::buffer_blas {

template <typename T>
MATHPRIM_BACKEND_BLAS_COPY_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  if constexpr (std::is_same_v<T, float>) {
    blasint inc_x = dst.stride(-1);
    blasint inc_y = src.stride(-1);
    blasint numel = dst.numel();
    // cblas_dcopy(const blasint n, const double *x, const blasint incx, double
    // *y, const blasint incy)
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_SCAL_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  for (auto sub : x.shape()) {
    x(sub) *= alpha;
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_SWAP_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  for (auto sub : x.shape()) {
    ::std::swap(x(sub), y(sub));
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_AXPY_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  for (auto sub : x.shape()) {
    y(sub) += alpha * x(sub);
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_DOT_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  T v = 0;
  for (auto sub : x.shape()) {
    v += y(sub) * x(sub);
  }
  return v;
}

template <typename T>
MATHPRIM_BACKEND_BLAS_NORM_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  T v = 0;
  for (auto sub : x.shape()) {
    v += x(sub) * x(sub);
  }
  return v;
}

template <typename T>
MATHPRIM_BACKEND_BLAS_ASUM_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  T v = 0;
  for (auto sub : x.shape()) {
    v += std::abs(x(sub));
  }
  return v;
}

template <typename T>
MATHPRIM_BACKEND_BLAS_AMAX_IMPL(T, blas_t::cpu_blas, device_t::cpu) {
  T v = 0;
  for (auto sub : x.shape()) {
    T xi = std::abs(x(sub));
    v = xi > v ? xi : v;
  }

  return v;
}

}  // namespace mathprim::buffer_blas
