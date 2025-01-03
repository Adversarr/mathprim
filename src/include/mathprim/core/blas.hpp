#pragma once
#include <stdexcept>

#include "mathprim/core/blas/basic_blas.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim::blas {

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void copy(basic_buffer_view<T, 1, dev> dst,
          basic_buffer_view<const T, 1, dev> src, blas_t = {}) {
  backend_blas<T, blas_t, dev>::copy(dst, src);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void scal(T alpha, basic_buffer_view<T, 1, dev> x, blas_t = {}) {
  backend_blas<T, blas_t, dev>::scal(alpha, x);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void swap(basic_buffer_view<T, 1, dev> x, basic_buffer_view<T, 1, dev> y,
          blas_t = {}) {
  backend_blas<T, blas_t, dev>::swap(x, y);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void axpy(T alpha, basic_buffer_view<const T, 1, dev> x,
          basic_buffer_view<T, 1, dev> y, blas_t = {}) {
  backend_blas<T, blas_t, dev>::axpy(alpha, x, y);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void gemv(T alpha, basic_buffer_view<const T, 2, dev> A,
          basic_buffer_view<const T, 1, dev> x, T beta,
          basic_buffer_view<T, 1, dev> y, blas_t = {}) {
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

  backend_blas<T, blas_t, dev>::gemv(alpha, A, x, beta, y);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void gemm(T alpha, basic_buffer_view<const T, 2, dev> A,
          basic_buffer_view<const T, 2, dev> B, T beta,
          basic_buffer_view<T, 2, dev> C, blas_t = {}) {
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
  backend_blas<T, blas_t, dev>::gemm(alpha, A, B, beta, C);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void emul(basic_buffer_view<const T, 1, dev> x, basic_buffer_view<T, 1, dev> y,
          blas_t = {}) {
  backend_blas<T, blas_t, dev>::emul(x, y);
};

template <typename T, device_t dev, typename blas_t = blas_select_t<dev>>
void ediv(basic_buffer_view<const T, 1, dev> x, basic_buffer_view<T, 1, dev> y,
          blas_t = {}) {
  backend_blas<T, blas_t, dev>::ediv(x, y);
};

}  // namespace mathprim::blas
