#pragma once

#include <cmath>
#include "mathprim/core/blas/utils.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim {
namespace blas {
namespace internal {}  // namespace internal

template <typename T> struct blas_impl_cpu_eigen {
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
void blas_impl_cpu_eigen<T>::copy(vector_view dst, const_vector_view src) {
  if (dst.is_contiguous() && src.is_contiguous()) {
    // faster copy
    auto dst_eigen = eigen_support::cmap(dst);
    auto src_eigen = eigen_support::cmap(src);
    dst_eigen = src_eigen;
  } else {
    auto dst_eigen = eigen_support::map(dst);
    auto src_eigen = eigen_support::map(src);
    dst_eigen = src_eigen;
  }
}
template <typename T>
void blas_impl_cpu_eigen<T>::scal(T alpha, vector_view x) {
  if (x.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    x_eigen *= alpha;
  } else {
    auto x_eigen = eigen_support::map(x);
    x_eigen *= alpha;
  }
}

template <typename T>
void blas_impl_cpu_eigen<T>::swap(vector_view x, vector_view y) {
  if (x.is_contiguous() && y.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    auto y_eigen = eigen_support::cmap(y);
    x_eigen.swap(y_eigen);
  } else {
    auto x_eigen = eigen_support::map(x);
    auto y_eigen = eigen_support::map(y);
    x_eigen.swap(y_eigen);
  }
}

template <typename T>
void blas_impl_cpu_eigen<T>::axpy(T alpha, const_vector_view x, vector_view y) {
  if (x.is_contiguous() && y.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    auto y_eigen = eigen_support::cmap(y);
    y_eigen += alpha * x_eigen;
  } else {
    auto x_eigen = eigen_support::map(x);
    auto y_eigen = eigen_support::map(y);
    y_eigen += alpha * x_eigen;
  }
}

template <typename T>
T blas_impl_cpu_eigen<T>::dot(const_vector_view x, const_vector_view y) {
  if (x.is_contiguous() && y.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    auto y_eigen = eigen_support::cmap(y);
    return x_eigen.dot(y_eigen);
  } else {
    auto x_eigen = eigen_support::map(x);
    auto y_eigen = eigen_support::map(y);
    return x_eigen.dot(y_eigen);
  }
}

template <typename T>
T blas_impl_cpu_eigen<T>::norm(const_vector_view x) {
  if (x.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    return x_eigen.norm();
  } else {
    auto x_eigen = eigen_support::map(x);
    return x_eigen.norm();
  }
}

template <typename T>
T blas_impl_cpu_eigen<T>::asum(const_vector_view x) {
  if (x.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    return x_eigen.cwiseAbs().sum();
  } else {
    auto x_eigen = eigen_support::map(x);
    return x_eigen.cwiseAbs().sum();
  }
}

template <typename T>
T blas_impl_cpu_eigen<T>::amax(const_vector_view x) {
  if (x.is_contiguous()) {
    auto x_eigen = eigen_support::cmap(x);
    return x_eigen.cwiseAbs().maxCoeff();
  } else {
    auto x_eigen = eigen_support::map(x);
    return x_eigen.cwiseAbs().maxCoeff();
  }
}

template <typename T>
void blas_impl_cpu_eigen<T>::gemv(T alpha, const_matrix_view A, const_vector_view x, T beta, vector_view y) {
  internal::check_mv_shapes(A.shape(), x.shape(), y.shape());
  if (y.is_contiguous()) {
    auto y_eigen = eigen_support::cmap(y);
    y_eigen *= beta;
    if (A.is_contiguous() && x.is_contiguous()) {
      auto A_eigen = eigen_support::cmap(A);
      auto x_eigen = eigen_support::cmap(x);
      y_eigen += alpha * A_eigen * x_eigen;
    } else {
      auto A_eigen = eigen_support::map(A);
      auto x_eigen = eigen_support::map(x);
      y_eigen += alpha * A_eigen * x_eigen;
    }
  } else {
    auto y_eigen = eigen_support::map(y);
    y_eigen *= beta;
    if (A.is_contiguous() && x.is_contiguous()) {
      auto A_eigen = eigen_support::cmap(A);
      auto x_eigen = eigen_support::cmap(x);
      y_eigen += alpha * A_eigen * x_eigen;
    } else {
      auto A_eigen = eigen_support::map(A);
      auto x_eigen = eigen_support::map(x);
      y_eigen += alpha * A_eigen * x_eigen;
    }
  }
}

template <typename T>
void blas_impl_cpu_eigen<T>::gemm(T alpha, const_matrix_view A, const_matrix_view B, T beta, matrix_view C) {
  internal::check_mm_shapes(A.shape(), B.shape(), C.shape());
  if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
    auto C_eigen = eigen_support::cmap(C);
    C_eigen *= beta;
    auto A_eigen = eigen_support::cmap(A);
    auto B_eigen = eigen_support::cmap(B);
    C_eigen += alpha * A_eigen * B_eigen;
  } else {
    auto C_eigen = eigen_support::map(C);
    C_eigen *= beta;
    auto A_eigen = eigen_support::map(A);
    auto B_eigen = eigen_support::map(B);
    C_eigen += alpha * A_eigen * B_eigen;
  }
}
}