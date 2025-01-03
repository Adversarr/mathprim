#pragma once

#include <cmath>

#include "basic_blas.hpp"
#include "helper_macros.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim {
namespace blas {

template <typename T>
struct backend_blas<T, blas_cpu_handmade, device_t::cpu> {
  static constexpr device_t dev = device_t::cpu;
  using vector_view = basic_buffer_view<T, 1, dev>;
  using matrix_view = basic_buffer_view<T, 2, dev>;
  using const_vector_view = basic_buffer_view<const T, 1, dev>;
  using const_matrix_view = basic_buffer_view<const T, 2, dev>;

  // Level 1
  static void copy(vector_view dst, const_vector_view src);
  static void scal(T alpha, vector_view x);
  static void swap(vector_view x, vector_view y);
  static void axpy(T alpha, const_vector_view x, vector_view y);

  static T dot(const_vector_view x, const_vector_view y);
  static T norm(const_vector_view x);
  static T asum(const_vector_view x);
  static T amax(const_vector_view x);

  // Level 2
  // y <- alpha * A * x + beta * y
  static void gemv(T alpha, const_matrix_view A, const_vector_view x, T beta,
                   vector_view y);

  // Level 3
  // C <- alpha * A * B + beta * C
  static void gemm(T alpha, const_matrix_view A, const_matrix_view B, T beta,
                   matrix_view C);

  // element-wise operatons
  // y = x * y
  static void emul(const_vector_view x, vector_view y);
  // y = x / y
  static void ediv(const_vector_view x, vector_view y);
};

}  // namespace blas

// Set default fallback.
template <>
struct blas_select_fallback<device_t::cpu> {
  using type = mathprim::MATHPRIM_INTERNAL_CPU_BLAS_FALLBACK;
};

}  // namespace mathprim

namespace mathprim::blas {

template <typename T>
MATHPRIM_BACKEND_BLAS_COPY_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t n = dst.numel();
  for (index_t i = 0; i < n; i++) {
    dst(i) = src(i);
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_SCAL_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    x(i) *= alpha;
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_SWAP_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    ::std::swap(x(i), y(i));
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_AXPY_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    y(i) += alpha * x(i);
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_DOT_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    v += y(i) * x(i);
  }
  return v;
}

template <typename T>
MATHPRIM_BACKEND_BLAS_NORM_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    v += x(i) * x(i);
  }
  return std::sqrt(v);
}

template <typename T>
MATHPRIM_BACKEND_BLAS_ASUM_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    v += std::abs(x(i));
  }
  return v;
}

template <typename T>
MATHPRIM_BACKEND_BLAS_AMAX_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    T xi = std::abs(x(i));
    v = xi > v ? xi : v;
  }
  return v;
}

template <typename T>
MATHPRIM_BACKEND_BLAS_GEMV_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t m = A.shape(0);
  index_t n = A.shape(1);
  MATHPRIM_ASSERT(x.shape(0) == n);
  MATHPRIM_ASSERT(y.shape(0) == m);
  for (index_t i = 0; i < m; i++) {
    T v = 0;
    for (index_t j = 0; j < n; j++) {
      v += A(i, j) * x(j);
    }
    y(i) = alpha * v + beta * y(i);
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_GEMM_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t m = A.shape(0);
  index_t n = B.shape(1);
  index_t k = A.shape(1);
  MATHPRIM_ASSERT(B.shape(0) == k);
  MATHPRIM_ASSERT(C.shape(0) == m);
  MATHPRIM_ASSERT(C.shape(1) == n);
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      T v = 0;
      for (index_t l = 0; l < k; l++) {
        v += A(i, l) * B(l, j);
      }
      C(i, j) = alpha * v + beta * C(i, j);
    }
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_EMUL_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    y(i) *= x(i);
  }
}

template <typename T>
MATHPRIM_BACKEND_BLAS_EDIV_IMPL(T, blas_cpu_handmade, device_t::cpu) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    y(i) /= x(i);
  }
}

}  // namespace mathprim::blas
