#pragma once

#include <cmath>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/common.hpp"
#include "utils.hpp"

namespace mathprim {
namespace blas {

template <typename T> struct blas_impl_cpu_handmade {
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

}  // namespace mathprim

namespace mathprim::blas {

template <typename T>
void blas_impl_cpu_handmade<T>::copy(vector_view dst, const_vector_view src) {
  index_t n = dst.numel();
  for (index_t i = 0; i < n; i++) {
    dst(i) = src(i);
  }
}

template <typename T>
void blas_impl_cpu_handmade<T>::scal(T alpha, vector_view x) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    x(i) *= alpha;
  }
}

template <typename T>
void blas_impl_cpu_handmade<T>::swap(vector_view x, vector_view y) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    ::std::swap(x(i), y(i));
  }
}

template <typename T>
void blas_impl_cpu_handmade<T>::axpy(T alpha, const_vector_view x,
                                     vector_view y) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    y(i) += alpha * x(i);
  }
}

template <typename T>
T blas_impl_cpu_handmade<T>::dot(const_vector_view x, const_vector_view y) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    v += y(i) * x(i);
  }
  return v;
}

template <typename T> T blas_impl_cpu_handmade<T>::norm(const_vector_view x) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    v += x(i) * x(i);
  }
  return std::sqrt(v);
}

template <typename T> T blas_impl_cpu_handmade<T>::asum(const_vector_view x) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    v += std::abs(x(i));
  }
  return v;
}

template <typename T> T blas_impl_cpu_handmade<T>::amax(const_vector_view x) {
  T v = 0;
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    T xi = std::abs(x(i));
    v = std::max(v, xi);
  }
  return v;
}

template <typename T>
void blas_impl_cpu_handmade<T>::gemv(T alpha, const_matrix_view A,
                                     const_vector_view x, T beta,
                                     vector_view y) {
  internal::check_mv_shapes(A.shape(), x.shape(), y.shape());

  index_t m = A.shape(0);
  index_t n = A.shape(1);
  // Optional: Add a configurable threshold for large matrices
  constexpr index_t threshold = 20;
  if (m * n > threshold) {
    // For better performance on large matrices, consider using a more optimized
    // BLAS library like OpenBLAS or Intel MKL.
    MATHPRIM_WARN_ONCE("cpu_handmade gemv is not optimized for large matrices");
  }
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
void blas_impl_cpu_handmade<T>::gemm(T alpha, const_matrix_view A,
                                     const_matrix_view B, T beta,
                                     matrix_view C) {
  internal::check_mm_shapes(A.shape(), B.shape(), C.shape());
  index_t m = A.shape(0);
  index_t n = B.shape(1);
  index_t k = A.shape(1);
  MATHPRIM_ASSERT(B.shape(0) == k);
  MATHPRIM_ASSERT(C.shape(1) == n);

  // Warning: cpu_handmade gemm is not optimized for large matrices.
  // This implementation is intended for small matrices due to its simplicity
  // and lack of optimizations. For better performance with large matrices,
  // consider using a more optimized BLAS library such as OpenBLAS or Intel MKL.
  MATHPRIM_WARN_ONCE("cpu_handmade gemm is not optimized for large matrices");
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < n; j++) {
      C(i, j) *= beta;
    }
  }

  const index_t block_size = 16;  // Block size for cache optimization
  for (index_t ii = 0; ii < m; ii += block_size) {
    for (index_t ll = 0; ll < k; ll += block_size) {
      for (index_t jj = 0; jj < n; jj += block_size) {
        for (index_t i = ii; i < std::min(ii + block_size, m); i++) {
          for (index_t l = ll; l < std::min(ll + block_size, k); l++) {
            T a_il = A(i, l);
            for (index_t j = jj; j < std::min(jj + block_size, n); j++) {
              C(i, j) += alpha * a_il * B(l, j);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void blas_impl_cpu_handmade<T>::emul(const_vector_view x, vector_view y) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    y(i) *= x(i);
  }
}

template <typename T>
void blas_impl_cpu_handmade<T>::ediv(const_vector_view x, vector_view y) {
  index_t n = x.numel();
  for (index_t i = 0; i < n; i++) {
    y(i) /= x(i);
  }
}

}  // namespace mathprim::blas
