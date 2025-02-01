#pragma once
#ifndef MATHPRIM_ENABLE_BLAS
#  error "BLAS is not enabled."
#endif

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/defines.hpp"
#ifdef MATHPRIM_BLAS_VENDOR_APPLE
#  include <Accelerate/Accelerate.h>
#  define CBLAS_INT CBLAS_INDEX
#  define CBLAS_LAYOUT CBLAS_ORDER
#else
#  include <cblas.h>
#endif

#include <cmath>
#include <type_traits>

#include "mathprim/blas/cpu_handmade.hpp"  // IWYU pragma: export
#include "mathprim/core/utils/common.hpp"
namespace mathprim::blas {
namespace internal {

constexpr CBLAS_TRANSPOSE invert(CBLAS_TRANSPOSE from) {
  return from == CblasNoTrans ? CblasTrans : CblasNoTrans;
}

template <typename Scalar, typename sshape, typename sstride, typename device>
constexpr CBLAS_INDEX vec_stride(basic_view<Scalar, sshape, sstride, device> view) {
  return view.stride(-1) / static_cast<index_t>(sizeof(Scalar));
}

constexpr CBLAS_TRANSPOSE to_blas(matrix_op mat_op) {
  return mat_op == matrix_op::none ? CblasNoTrans : CblasTrans;
}

}  // namespace internal

template <typename T>
struct cpu_blas : public basic_blas<cpu_blas<T>, T, device::cpu> {
  // Level 1
  template <typename sshape, typename sstride>
  using view_type = basic_view<T, sshape, sstride, device::cpu>;
  template <typename sshape, typename sstride>
  using const_type = basic_view<const T, sshape, sstride, device::cpu>;

  using Scalar = T;

  template <typename sshape_dst, typename sstride_dst, typename sshape_src, typename sstride_src>
  void copy_impl(view_type<sshape_dst, sstride_dst> dst, const_type<sshape_src, sstride_src> src) {
    auto dst_stride = internal::vec_stride(dst);
    auto src_stride = internal::vec_stride(src);
    CBLAS_INT numel = static_cast<CBLAS_INT>(dst.numel());

    if constexpr (std::is_same_v<Scalar, float>) {
      cblas_scopy(numel, src.data(), src_stride, dst.data(), dst_stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      cblas_dcopy(numel, src.data(), src_stride, dst.data(), dst_stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS copy.");
    }
  }

  template <typename sshape, typename sstride>
  void scal_impl(T alpha, view_type<sshape, sstride> src) {
    auto stride = internal::vec_stride(src);
    CBLAS_INT numel = static_cast<CBLAS_INT>(src.numel());

    if constexpr (std::is_same_v<Scalar, float>) {
      cblas_sscal(numel, alpha, src.data(), stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      cblas_dscal(numel, alpha, src.data(), stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS scal.");
    }
  }

  template <typename sshape_src, typename sstride_src, typename sshape_dst, typename sstride_dst>
  void swap_impl(view_type<sshape_src, sstride_src> src, view_type<sshape_dst, sstride_dst> dst) {
    // auto shape = src.shape();
    // for (auto sub : shape) {
    //   ::std::swap(src(sub), dst(sub));
    // }
    auto src_stride = internal::vec_stride(src);
    auto dst_stride = internal::vec_stride(dst);
    auto numel = static_cast<CBLAS_INT>(src.numel());

    if constexpr (std::is_same_v<Scalar, float>) {
      cblas_sswap(numel, src.data(), src_stride, dst.data(), dst_stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      cblas_dswap(numel, src.data(), src_stride, dst.data(), dst_stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS swap.");
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  void axpy_impl(T alpha, const_type<sshape_x, sstride_x> x, view_type<sshape_y, sstride_y> y) {
    auto x_stride = internal::vec_stride(x);
    auto y_stride = internal::vec_stride(y);
    auto numel = static_cast<CBLAS_INT>(x.numel());

    if constexpr (std::is_same_v<Scalar, float>) {
      cblas_saxpy(numel, alpha, x.data(), x_stride, y.data(), y_stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      cblas_daxpy(numel, alpha, x.data(), x_stride, y.data(), y_stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS axpy.");
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  T dot_impl(const_type<sshape_x, sstride_x> x, const_type<sshape_y, sstride_y> y) {
    auto x_stride = internal::vec_stride(x);
    auto y_stride = internal::vec_stride(y);
    auto numel = static_cast<CBLAS_INT>(x.numel());

    if constexpr (std::is_same_v<Scalar, float>) {
      return cblas_sdot(numel, x.data(), x_stride, y.data(), y_stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      return cblas_ddot(numel, x.data(), x_stride, y.data(), y_stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS dot.");
    }
  }

  template <typename sshape, typename sstride>
  T norm_impl(const_type<sshape, sstride> x) {
    auto stride = internal::vec_stride(x);
    auto numel = static_cast<CBLAS_INT>(x.numel());
    Scalar out = 0;
    if constexpr (std::is_same_v<Scalar, float>) {
      out = cblas_snrm2(numel, x.data(), stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      out = cblas_dnrm2(numel, x.data(), stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS norm.");
    }
    return out;
  }

  template <typename sshape, typename sstride>
  T asum_impl(const_type<sshape, sstride> x) {
    auto stride = internal::vec_stride(x);
    auto numel = static_cast<CBLAS_INT>(x.numel());
    Scalar out = 0;
    if constexpr (std::is_same_v<Scalar, float>) {
      out = cblas_sasum(numel, x.data(), stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      out = cblas_dasum(numel, x.data(), stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS asum.");
    }
    return out;
  }

  template <typename sshape, typename sstride>
  index_t amax_impl(const_type<sstride, sshape> x) {
    auto stride = internal::vec_stride(x);
    auto numel = static_cast<CBLAS_INT>(x.numel());
    index_t out = 0;
    if constexpr (std::is_same_v<Scalar, float>) {
      out = cblas_isamax(numel, x.data(), stride);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      out = cblas_idamax(numel, x.data(), stride);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS amax.");
    }
    return out;
  }

  // element-wise operatons
  // y = x * y
  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  void emul_impl(T alpha, const_type<sshape_x, sstride_x> x, T beta, view_type<sshape_y, sstride_y> y) {
    auto shape = x.shape();
    for (auto sub : shape) {
      y(sub) = y(sub) * beta + y(sub) * alpha * x(sub);
    }
  }
  // y = x / y
  template <typename sshape_x, typename sstride_x, typename sshape_y, typename sstride_y>
  void ediv_impl(T alpha, const_type<sshape_x, sstride_x> x, T beta, view_type<sshape_y, sstride_y> y) {
    auto shape = x.shape();
    for (auto sub : shape) {
      y(sub) = y(sub) / beta + y(sub) / alpha / x(sub);
    }
  }

  // // Level 2
  // // y <- alpha * A * x + beta * y
  template <typename sshape_A, typename sstride_A, typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  void gemv_impl(T alpha, const_type<sshape_A, sstride_A> A, const_type<sshape_x, sstride_x> x, T beta,
                 view_type<sshape_y, sstride_y> y) {
    auto [n, m] = A.shape();
    // it always computes A x, but A could be a transpose of some other matrix
    auto mat_op = internal::to_blas(internal::get_matrix_op(A));
    CBLAS_INT lda = 0;
    if (mat_op == CblasNoTrans) {
      lda = A.stride(-2) / static_cast<index_t>(sizeof(T));
    } else {
      lda = A.stride(-1) / static_cast<index_t>(sizeof(T));
      std::swap(n, m);  // Transpose the matrix.
    }

    CBLAS_INT incx = x.stride(-1) / static_cast<index_t>(sizeof(T));
    CBLAS_INT incy = y.stride(-1) / static_cast<index_t>(sizeof(T));
    if constexpr (std::is_same_v<Scalar, float>) {
      cblas_sgemv(CblasRowMajor, mat_op, n, m, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy);
    } else if constexpr (std::is_same_v<Scalar, double>) {
      cblas_dgemv(CblasRowMajor, mat_op, n, m, alpha, A.data(), lda, x.data(), incx, beta, y.data(), incy);
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS amax.");
    }
  }

  //
  // // Level 3
  // // C <- alpha * A * B + beta * C
  template <typename sshape_A, typename sstride_A, typename sshape_B, typename sstride_B, typename sshape_C,
            typename sstride_C>
  void gemm_impl(T alpha, const_type<sshape_A, sstride_A> A, const_type<sshape_B, sstride_B> B, T beta,
                 view_type<sshape_C, sstride_C> C) {
    auto mat_op_C = internal::get_matrix_op(C);
    if (mat_op_C == internal::matrix_op::transpose) {
      gemm_impl(alpha, B.transpose(), A.transpose(), beta, C.transpose());
      return;
    }

    // c must not be a transposed view
    auto [m, n] = C.shape();

    auto mat_op_A = internal::to_blas(internal::get_matrix_op(A));
    auto mat_op_B = internal::to_blas(internal::get_matrix_op(B));
    CBLAS_INT lda = 0;
    CBLAS_INT ldb = 0;
    CBLAS_INT ldc = C.stride(-2) / static_cast<index_t>(sizeof(T));

    if (mat_op_A == CblasNoTrans) {
      lda = A.stride(-2) / static_cast<index_t>(sizeof(T));
    } else {
      lda = A.stride(-1) / static_cast<index_t>(sizeof(T));
    }

    if (mat_op_B == CblasNoTrans) {
      ldb = B.stride(-2) / static_cast<index_t>(sizeof(T));
    } else {
      ldb = B.stride(-1) / static_cast<index_t>(sizeof(T));
    }

    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, mat_op_A, mat_op_B, m, n, A.size(-1), alpha, A.data(), lda, B.data(), ldb, beta,
                  C.data(), ldc);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, mat_op_A, mat_op_B, m, n, A.size(-1), alpha, A.data(), lda, B.data(), ldb, beta,
                  C.data(), ldc);
    } else {
      static_assert(::mathprim::internal::always_false_v<T>, "Unsupported type for BLAS gemm.");
    }
  }
};

// template <typename T>
// struct blas_impl_cpu_blas {
//   static constexpr device_t dev = device_t::cpu;
//   using blas_unsupported = blas_impl_cpu_handmade<T>;
//   using vector_view = basic_view<T, 1, dev>;
//   using matrix_view = basic_view<T, 2, dev>;
//   using const_vector_view = basic_view<const T, 1, dev>;
//   using const_matrix_view = basic_view<const T, 2, dev>;
//
//   // Level 1
//   static void copy(vector_view dst, const_vector_view src);
//   static void scal(T alpha, vector_view x);
//   static void swap(vector_view x, vector_view y);
//   static void axpy(T alpha, const_vector_view x, vector_view y);
//
//   static T dot(const_vector_view x, const_vector_view y);
//   static T norm(const_vector_view x);
//   static T asum(const_vector_view x);
//   static T amax(const_vector_view x);
//
//   // Level 2
//   // y <- alpha * A * x + beta * y
//   static void gemv(T alpha, const_matrix_view A, const_vector_view x, T beta, vector_view y);
//
//   // Level 3
//   // C <- alpha * A * B + beta * C
//   static void gemm(T alpha, const_matrix_view A, const_matrix_view B, T beta, matrix_view C);
//
//   // element-wise operatons
//   // y = x * y
//   static void emul(const_vector_view x, vector_view y);
//   // y = x / y
//   static void ediv(const_vector_view x, vector_view y);
// };
//
// }  // namespace blas
//
// }  // namespace mathprim
//
// namespace mathprim::blas {
//
// template <typename T>
// void blas_impl_cpu_blas<T>::copy(vector_view dst, const_vector_view src) {
//   CBLAS_INT inc_x = dst.stride(-1);
//   CBLAS_INT inc_y = src.stride(-1);
//   CBLAS_INT numel = dst.numel();
//   if constexpr (std::is_same_v<T, float>) {
//     cblas_scopy(numel, src.data(), inc_y, dst.data(), inc_x);
//   } else if constexpr (std::is_same_v<T, double>) {
//     cblas_dcopy(numel, src.data(), inc_y, dst.data(), inc_x);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS copy.");
//     blas_unsupported::copy(dst, src);
//   }
// }
//
// template <typename T>
// void blas_impl_cpu_blas<T>::scal(T alpha, vector_view x) {
//   CBLAS_INT inc_x = x.stride(-1);
//   CBLAS_INT numel = x.numel();
//   T* x_data = x.data();
//
//   if constexpr (std::is_same_v<T, float>) {
//     cblas_sscal(numel, alpha, x_data, inc_x);
//   } else if constexpr (std::is_same_v<T, double>) {
//     cblas_dscal(numel, alpha, x_data, inc_x);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS scal.");
//     blas_unsupported::scal(alpha, x);
//   }
// }
//
// template <typename T>
// void blas_impl_cpu_blas<T>::swap(vector_view x, vector_view y) {
//   CBLAS_INT inc_x = x.stride(-1);
//   CBLAS_INT inc_y = y.stride(-1);
//   CBLAS_INT numel = x.numel();
//   T* x_data = x.data();
//   T* y_data = y.data();
//
//   if constexpr (std::is_same_v<T, float>) {
//     cblas_sswap(numel, x_data, inc_x, y_data, inc_y);
//   } else if constexpr (std::is_same_v<T, double>) {
//     cblas_dswap(numel, x_data, inc_x, y_data, inc_y);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS swap.");
//     blas_unsupported::swap(x, y);
//   }
// }
//
// template <typename T>
// void blas_impl_cpu_blas<T>::axpy(T alpha, const_vector_view x, vector_view y) {
//   CBLAS_INT inc_x = x.stride(-1);
//   CBLAS_INT inc_y = y.stride(-1);
//   CBLAS_INT numel = x.numel();
//   const T* x_data = x.data();
//   T* y_data = y.data();
//
//   if constexpr (std::is_same_v<T, float>) {
//     cblas_saxpy(numel, alpha, x_data, inc_x, y_data, inc_y);
//   } else if constexpr (std::is_same_v<T, double>) {
//     cblas_daxpy(numel, alpha, x_data, inc_x, y_data, inc_y);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS axpy.");
//     blas_unsupported::axpy(alpha, x, y);
//   }
// }
//
// template <typename T>
// void blas_impl_cpu_blas<T>::emul(const_vector_view x, vector_view y) {
//   T v = 0;
//   CBLAS_INT inc_x = x.stride(-1);
//   CBLAS_INT inc_y = y.stride(-1);
//   CBLAS_INT numel = x.numel();
//   const T* x_data = x.data();
//   const T* y_data = y.data();
//
//   if constexpr (std::is_same_v<T, float>) {
//     v = cblas_sdot(numel, x_data, inc_x, y_data, inc_y);
//   } else if constexpr (std::is_same_v<T, double>) {
//     v = cblas_ddot(numel, x_data, inc_x, y_data, inc_y);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS dot.");
//     blas_unsupported::dot(x, y);
//   }
//   return v;
// }
//
// template <typename T>
// T blas_impl_cpu_blas<T>::norm(const_vector_view x) {
//   T v = 0;
//   CBLAS_INT inc_x = x.stride(-1);
//   if constexpr (std::is_same_v<T, float>) {
//     v = cblas_snrm2(x.numel(), x.data(), inc_x);
//   } else if constexpr (std::is_same_v<T, double>) {
//     v = cblas_dnrm2(x.numel(), x.data(), inc_x);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS norm.");
//     blas_unsupported::norm(x);
//   }
//   return v;
// }
//
// template <typename T>
// T blas_impl_cpu_blas<T>::asum(const_vector_view x) {
//   T v = 0;
//   CBLAS_INT inc_x = x.stride(-1);
//   if constexpr (std::is_same_v<T, float>) {
//     v = cblas_sasum(x.numel(), x.data(), inc_x);
//   } else if constexpr (std::is_same_v<T, double>) {
//     v = cblas_dasum(x.numel(), x.data(), inc_x);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS asum.");
//     blas_unsupported::asum(x);
//   }
//   return v;
// }
//
// template <typename T>
// T blas_impl_cpu_blas<T>::amax(const_vector_view x) {
//   T v = 0;
//   CBLAS_INT inc_x = x.stride(-1);
//   if constexpr (std::is_same_v<T, float>) {
//     v = cblas_isamax(x.numel(), x.data(), inc_x);
//   } else if constexpr (std::is_same_v<T, double>) {
//     v = cblas_idamax(x.numel(), x.data(), inc_x);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS amax.");
//     blas_unsupported::amax(x);
//   }
//   return v;
// }
//
// template <typename T>
// void blas_impl_cpu_blas<T>::gemv(T alpha, const_matrix_view A, const_vector_view x, T beta, vector_view y) {
//   internal::check_mv_shapes(A.shape(), x.shape(), y.shape());
//
//   CBLAS_INT m = A.size(0);
//   CBLAS_INT n = A.size(1);
//   CBLAS_INT row_stride = A.stride(0);
//   CBLAS_INT col_stride = A.stride(1);
//   CBLAS_INT inc_x = x.stride(0);
//   CBLAS_INT inc_y = y.stride(0);
//
//   CBLAS_TRANSPOSE transpose_op;
//   CBLAS_INT lda = 0;
//   if (row_stride > 1 && col_stride == 1) {
//     transpose_op = CblasNoTrans;
//     lda = row_stride;
//   } else if (row_stride == 1 && col_stride > 1) {
//     // is a transpose view of some matrix, so we need to transpose back and use
//     // the original matrix
//     transpose_op = CblasTrans;
//     lda = col_stride;
//     std::swap(m, n);
//   } else {
//     MATHPRIM_INTERNAL_FATAL("Invalid matrix view for BLAS gemv.");
//   }
//
//   if constexpr (std::is_same_v<T, float>) {
//     cblas_sgemv(CblasRowMajor, transpose_op, m, n, alpha, A.data(), lda, x.data(), inc_x, beta, y.data(), inc_y);
//   } else if constexpr (std::is_same_v<T, double>) {
//     cblas_dgemv(CblasRowMajor, transpose_op, m, n, alpha, A.data(), lda, x.data(), inc_x, beta, y.data(), inc_y);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS gemv.");
//     blas_unsupported::gemv(alpha, A, x, beta, y);
//   }
// }
//
// template <typename T>
// void blas_impl_cpu_blas<T>::gemm(T alpha, const_matrix_view A, const_matrix_view B, T beta, matrix_view C) {
//   internal::check_mm_shapes(A.shape(), B.shape(), C.shape());
//
//   CBLAS_INT m = C.size(0);
//   CBLAS_INT n = C.size(1);
//   CBLAS_INT k = A.size(1);
//   CBLAS_INT lda = A.stride(0), lda_t = A.stride(1);
//   CBLAS_INT ldb = B.stride(0), ldb_t = B.stride(1);
//   CBLAS_INT ldc = C.stride(0), ldc_t = C.stride(1);
//
//   // First check C, if it is a transposed view, we view it as ColMajor.
//   if (ldc_t > 1 && ldc == 1) {
//     // do equivalent operation on C.T
//     // C.T <- alpha * B.T * A.T + beta * C.T
//     gemm(alpha, B.transpose(0, 1), A.transpose(0, 1), beta, C.transpose(0, 1));
//     return;
//   }
//
//   CBLAS_TRANSPOSE tr_a = CblasNoTrans, tr_b = CblasNoTrans;
//   const CBLAS_LAYOUT lay_c = CblasRowMajor;
//   const T *left_matrix = A.data(), *right_matrix = B.data();
//
//   if (lda == 1) {
//     tr_a = CblasTrans;
//     lda = lda_t;
//     // m, k indicates op(A).
//   }
//
//   if (ldb == 1) {
//     tr_b = CblasTrans;
//     ldb = ldb_t;
//     // k, n indicates op(B).
//   }
//
//   if constexpr (std::is_same_v<T, float>) {
//     cblas_sgemm(lay_c, tr_a, tr_b, m, n, k, alpha, left_matrix, lda, right_matrix, ldb, beta, C.data(), ldc);
//   } else if constexpr (std::is_same_v<T, double>) {
//     cblas_dgemm(lay_c, tr_a, tr_b, m, n, k, alpha, left_matrix, lda, right_matrix, ldb, beta, C.data(), ldc);
//   } else {
//     MATHPRIM_WARN_ONCE("Unsupported type for BLAS gemm.");
//     blas_unsupported::gemm(alpha, A, B, beta, C);
//   }
// }

}  // namespace mathprim::blas
