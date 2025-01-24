#pragma once
#ifndef MATHPRIM_ENABLE_BLAS
#  error "BLAS is not enabled."
#endif

#ifdef MATHPRIM_BLAS_VENDOR_APPLE
#include <Accelerate/Accelerate.h>
#define CBLAS_INT CBLAS_INDEX
#define CBLAS_LAYOUT CBLAS_ORDER
#else
#include <cblas.h>
#endif

#include <cmath>
#include <type_traits>

#include "mathprim/core/blas/cpu_handmade.hpp"  // IWYU pragma: export
#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/common.hpp"
#include "utils.hpp"

namespace mathprim {
namespace blas {
namespace internal {

constexpr CBLAS_TRANSPOSE invert(CBLAS_TRANSPOSE from) {
  return from == CblasNoTrans ? CblasTrans : CblasNoTrans;
}

}  // namespace internal

template <typename T> struct blas_impl_cpu_blas {
  static constexpr device_t dev = device_t::cpu;
  using blas_unsupported = blas_impl_cpu_handmade<T>;
  using vector_view = basic_view<T, 1, dev>;
  using matrix_view = basic_view<T, 2, dev>;
  using const_vector_view = basic_view<const T, 1, dev>;
  using const_matrix_view = basic_view<const T, 2, dev>;

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
void blas_impl_cpu_blas<T>::copy(vector_view dst, const_vector_view src) {
  CBLAS_INT inc_x = dst.stride(-1);
  CBLAS_INT inc_y = src.stride(-1);
  CBLAS_INT numel = dst.numel();
  if constexpr (std::is_same_v<T, float>) {
    cblas_scopy(numel, src.data(), inc_y, dst.data(), inc_x);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dcopy(numel, src.data(), inc_y, dst.data(), inc_x);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS copy.");
    blas_unsupported::copy(dst, src);
  }
}

template <typename T> void blas_impl_cpu_blas<T>::scal(T alpha, vector_view x) {
  CBLAS_INT inc_x = x.stride(-1);
  CBLAS_INT numel = x.numel();
  T* x_data = x.data();

  if constexpr (std::is_same_v<T, float>) {
    cblas_sscal(numel, alpha, x_data, inc_x);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dscal(numel, alpha, x_data, inc_x);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS scal.");
    blas_unsupported::scal(alpha, x);
  }
}

template <typename T>
void blas_impl_cpu_blas<T>::swap(vector_view x, vector_view y) {
  CBLAS_INT inc_x = x.stride(-1);
  CBLAS_INT inc_y = y.stride(-1);
  CBLAS_INT numel = x.numel();
  T* x_data = x.data();
  T* y_data = y.data();

  if constexpr (std::is_same_v<T, float>) {
    cblas_sswap(numel, x_data, inc_x, y_data, inc_y);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dswap(numel, x_data, inc_x, y_data, inc_y);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS swap.");
    blas_unsupported::swap(x, y);
  }
}

template <typename T>
void blas_impl_cpu_blas<T>::axpy(T alpha, const_vector_view x, vector_view y) {
  CBLAS_INT inc_x = x.stride(-1);
  CBLAS_INT inc_y = y.stride(-1);
  CBLAS_INT numel = x.numel();
  const T* x_data = x.data();
  T* y_data = y.data();

  if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(numel, alpha, x_data, inc_x, y_data, inc_y);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(numel, alpha, x_data, inc_x, y_data, inc_y);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS axpy.");
    blas_unsupported::axpy(alpha, x, y);
  }
}

template <typename T>
void blas_impl_cpu_blas<T>::emul(const_vector_view x, vector_view y) {
  T v = 0;
  CBLAS_INT inc_x = x.stride(-1);
  CBLAS_INT inc_y = y.stride(-1);
  CBLAS_INT numel = x.numel();
  const T* x_data = x.data();
  const T* y_data = y.data();

  if constexpr (std::is_same_v<T, float>) {
    v = cblas_sdot(numel, x_data, inc_x, y_data, inc_y);
  } else if constexpr (std::is_same_v<T, double>) {
    v = cblas_ddot(numel, x_data, inc_x, y_data, inc_y);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS dot.");
    blas_unsupported::dot(x, y);
  }
  return v;
}

template <typename T> T blas_impl_cpu_blas<T>::norm(const_vector_view x) {
  T v = 0;
  CBLAS_INT inc_x = x.stride(-1);
  if constexpr (std::is_same_v<T, float>) {
    v = cblas_snrm2(x.numel(), x.data(), inc_x);
  } else if constexpr (std::is_same_v<T, double>) {
    v = cblas_dnrm2(x.numel(), x.data(), inc_x);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS norm.");
    blas_unsupported::norm(x);
  }
  return v;
}

template <typename T> T blas_impl_cpu_blas<T>::asum(const_vector_view x) {
  T v = 0;
  CBLAS_INT inc_x = x.stride(-1);
  if constexpr (std::is_same_v<T, float>) {
    v = cblas_sasum(x.numel(), x.data(), inc_x);
  } else if constexpr (std::is_same_v<T, double>) {
    v = cblas_dasum(x.numel(), x.data(), inc_x);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS asum.");
    blas_unsupported::asum(x);
  }
  return v;
}

template <typename T> T blas_impl_cpu_blas<T>::amax(const_vector_view x) {
  T v = 0;
  CBLAS_INT inc_x = x.stride(-1);
  if constexpr (std::is_same_v<T, float>) {
    v = cblas_isamax(x.numel(), x.data(), inc_x);
  } else if constexpr (std::is_same_v<T, double>) {
    v = cblas_idamax(x.numel(), x.data(), inc_x);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS amax.");
    blas_unsupported::amax(x);
  }
  return v;
}

template <typename T>
void blas_impl_cpu_blas<T>::gemv(T alpha, const_matrix_view A,
                                 const_vector_view x, T beta, vector_view y) {
  internal::check_mv_shapes(A.shape(), x.shape(), y.shape());

  CBLAS_INT m = A.size(0);
  CBLAS_INT n = A.size(1);
  CBLAS_INT row_stride = A.stride(0);
  CBLAS_INT col_stride = A.stride(1);
  CBLAS_INT inc_x = x.stride(0);
  CBLAS_INT inc_y = y.stride(0);

  CBLAS_TRANSPOSE transpose_op;
  CBLAS_INT lda = 0;
  if (row_stride > 1 && col_stride == 1) {
    transpose_op = CblasNoTrans;
    lda = row_stride;
  } else if (row_stride == 1 && col_stride > 1) {
    // is a transpose view of some matrix, so we need to transpose back and use
    // the original matrix
    transpose_op = CblasTrans;
    lda = col_stride;
    std::swap(m, n);
  } else {
    MATHPRIM_INTERNAL_FATAL("Invalid matrix view for BLAS gemv.");
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemv(CblasRowMajor, transpose_op, m, n, alpha, A.data(), lda,
                x.data(), inc_x, beta, y.data(), inc_y);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemv(CblasRowMajor, transpose_op, m, n, alpha, A.data(), lda,
                x.data(), inc_x, beta, y.data(), inc_y);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS gemv.");
    blas_unsupported::gemv(alpha, A, x, beta, y);
  }
}

template <typename T>
void blas_impl_cpu_blas<T>::gemm(T alpha, const_matrix_view A,
                                 const_matrix_view B, T beta, matrix_view C) {
  internal::check_mm_shapes(A.shape(), B.shape(), C.shape());

  CBLAS_INT m = C.size(0);
  CBLAS_INT n = C.size(1);
  CBLAS_INT k = A.size(1);
  CBLAS_INT lda = A.stride(0), lda_t = A.stride(1);
  CBLAS_INT ldb = B.stride(0), ldb_t = B.stride(1);
  CBLAS_INT ldc = C.stride(0), ldc_t = C.stride(1);

  // First check C, if it is a transposed view, we view it as ColMajor.
  if (ldc_t > 1 && ldc == 1) {
    // do equivalent operation on C.T
    // C.T <- alpha * B.T * A.T + beta * C.T
    gemm(alpha, B.transpose(0, 1), A.transpose(0, 1), beta, C.transpose(0, 1));
    return;
  }

  CBLAS_TRANSPOSE tr_a = CblasNoTrans, tr_b = CblasNoTrans;
  const CBLAS_LAYOUT lay_c = CblasRowMajor;
  const T *left_matrix = A.data(), *right_matrix = B.data();

  if (lda == 1) {
    tr_a = CblasTrans;
    lda = lda_t;
    // m, k indicates op(A).
  }

  if (ldb == 1) {
    tr_b = CblasTrans;
    ldb = ldb_t;
    // k, n indicates op(B).
  }

  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(lay_c, tr_a, tr_b, m, n, k, alpha, left_matrix, lda,
                right_matrix, ldb, beta, C.data(), ldc);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(lay_c, tr_a, tr_b, m, n, k, alpha, left_matrix, lda,
                right_matrix, ldb, beta, C.data(), ldc);
  } else {
    MATHPRIM_WARN_ONCE("Unsupported type for BLAS gemm.");
    blas_unsupported::gemm(alpha, A, B, beta, C);
  }
}

}  // namespace mathprim::blas
