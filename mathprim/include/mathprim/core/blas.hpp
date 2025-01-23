#pragma once
#include "mathprim/core/defines.hpp"

namespace mathprim::blas {

// Assumption:
// template <typename T>
// struct blas_impl {
//   using vector_view = basic_view<T, 1, dev>;
//   using matrix_view = basic_view<T, 2, dev>;
//   using const_vector_view = basic_view<const T, 1, dev>;
//   using const_matrix_view = basic_view<const T, 2, dev>;
//   // Level 1
//   static void copy(vector_view dst, const_vector_view src);
//   static void scal(T alpha, vector_view x);
//   static void swap(vector_view x, vector_view y);
//   static void axpy(T alpha, const_vector_view x, vector_view y);

//   static T dot(const_vector_view x, const_vector_view y);
//   static T norm(const_vector_view x);
//   static T asum(const_vector_view x);
//   static T amax(const_vector_view x);
//   // Level 2
//   // y <- alpha * A * x + beta * y
//   static void gemv(T alpha, const_matrix_view A, const_vector_view x, T beta,
//             vector_view y);
//   // Level 3
//   // C <- alpha * A * B + beta * C
//   static void gemm(T alpha, const_matrix_view A, const_matrix_view B, T beta,
//             matrix_view C);
//   // element-wise operatons
//   // y = x * y
//   static void emul(const_vector_view x, vector_view y);
//   // y = x / y
//   static void ediv(const_vector_view x, vector_view y);
// };

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void copy(basic_view<T, 1, dev> dst,
          basic_view<const T, 1, dev> src, blas_impl = {}) {
  blas_impl::copy(dst, src);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void scal(T alpha, basic_view<T, 1, dev> x, blas_impl = {}) {
  blas_impl::scal(alpha, x);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void swap(basic_view<T, 1, dev> x, basic_view<T, 1, dev> y,
          blas_impl = {}) {
  blas_impl::swap(x, y);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void axpy(T alpha, basic_view<const T, 1, dev> x,
          basic_view<T, 1, dev> y, blas_impl = {}) {
  blas_impl::axpy(alpha, x, y);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void gemv(T alpha, basic_view<const T, 2, dev> A,
          basic_view<const T, 1, dev> x, T beta,
          basic_view<T, 1, dev> y, blas_impl = {}) {
  // check valid.
  index_t rows = A.size(0), cols = A.size(1);
  index_t lda = A.stride(0), lda_t = A.stride(1);
  index_t x_size = x.size(0), y_size = y.size(0);
  if (cols != x_size) {
    throw std::runtime_error(
        "Invalid matrix-vector view for BLAS gemv. (cols != x_size)");
  } else if (rows != y_size) {
    throw std::runtime_error(
        "Invalid matrix-vector view for BLAS gemv. (rows != y_size)");
  } else if (lda != 1 && lda_t != 1) {
    throw std::runtime_error(
        "Invalid matrix view, expected at least one stride to be 1.");
  }

  blas_impl::gemv(alpha, A, x, beta, y);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void gemm(T alpha, basic_view<const T, 2, dev> A,
          basic_view<const T, 2, dev> B, T beta,
          basic_view<T, 2, dev> C, blas_impl = {}) {
  index_t m = C.size(0), n = C.size(1), m2 = A.size(0), n2 = B.size(1);
  index_t k = A.size(1), k2 = B.size(0);
  index_t lda = A.stride(0), lda_t = A.stride(1);
  index_t ldb = B.stride(0), ldb_t = B.stride(1);
  index_t ldc = C.stride(0), ldc_t = C.stride(1);
  if (m != m2) {
    throw std::runtime_error(
        "Invalid matrix-matrix view for BLAS gemm. (m != m2)");
  } else if (n != n2) {
    throw std::runtime_error(
        "Invalid matrix-matrix view for BLAS gemm. (n != n2)");
  } else if (k != k2) {
    throw std::runtime_error(
        "Invalid matrix-matrix view for BLAS gemm. (k != k2)");
  } else if (lda != 1 && lda_t != 1) {
    throw std::runtime_error(
        "Invalid matrix view, expected at least one stride to be 1.");
  } else if (ldb != 1 && ldb_t != 1) {
    throw std::runtime_error(
        "Invalid matrix view, expected at least one stride to be 1.");
  } else if (ldc != 1 && ldc_t != 1) {
    throw std::runtime_error(
        "Invalid matrix view, expected at least one stride to be 1.");
  }
  blas_impl::gemm(alpha, A, B, beta, C);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void emul(basic_view<const T, 1, dev> x, basic_view<T, 1, dev> y,
          blas_impl = {}) {
  blas_impl::emul(x, y);
};

template <typename T, device_t dev, typename blas_impl = blas_select_t<T, dev>>
void ediv(basic_view<const T, 1, dev> x, basic_view<T, 1, dev> y,
          blas_impl = {}) {
  blas_impl::ediv(x, y);
};

}  // namespace mathprim::blas
