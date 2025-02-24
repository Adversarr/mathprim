#pragma once

#include <vector>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/defines.hpp"
#ifdef MATHPRIM_BLAS_VENDOR_APPLE
#  include <Accelerate/Accelerate.h>
#  ifndef CBLAS_INT
#    define CBLAS_INT int
#  endif
#elif (defined(MATHPRIM_BLAS_VENDOR_OPENBLAS))
#  include <openblas/cblas.h>
#  ifndef CBLAS_INT
#    define CBLAS_INT blasint
#  endif
#elif (defined(MATHPRIM_BLAS_VENDOR_INTEL_MKL))
#  include <mkl_cblas.h>
#define CBLAS_INT MKL_INT
#else
#  error "Unsupported BLAS vendor."
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
constexpr CBLAS_INT vec_stride(basic_view<Scalar, sshape, sstride, device> view) {
  return view.stride(-1);
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

  using base = basic_blas<cpu_blas<T>, T, device::cpu>;
  friend base;
  using Scalar = T;

protected:
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
  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  MATHPRIM_NOINLINE void emul_impl(const_type<SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
    auto total = x.shape(0);
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      y[i] = x[i] * y[i];
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
      lda = A.stride(-2);
    } else {
      lda = A.stride(-1);
      std::swap(n, m);  // Transpose the matrix.
    }

    CBLAS_INT incx = x.stride(-1);
    CBLAS_INT incy = y.stride(-1);
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
    CBLAS_INT ldc = C.stride(-2);

    if (mat_op_A == CblasNoTrans) {
      lda = A.stride(-2);
    } else {
      lda = A.stride(-1);
    }

    if (mat_op_B == CblasNoTrans) {
      ldb = B.stride(-2);
    } else {
      ldb = B.stride(-1);
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

  template <typename SshapeA, typename SstrideA, typename SshapeB, typename SstrideB, typename SshapeC,
            typename SstrideC>
  MATHPRIM_NOINLINE void gemm_batched_impl(Scalar alpha, const_type<SshapeA, SstrideA> A,
                                           const_type<SshapeB, SstrideB> B, Scalar beta,
                                           view_type<SshapeC, SstrideC> C) {
#if (defined(MATHPRIM_BLAS_VENDOR_INTEL_MKL))
    auto mat_op_C = internal::get_matrix_op(C.template slice<0>(0));
    if (mat_op_C == internal::matrix_op::transpose) {
      gemm_batched_impl(alpha, B.transpose(), A.transpose(), beta, C.transpose());
      return;
    }

    CBLAS_TRANSPOSE mat_op_A = internal::to_blas(internal::get_matrix_op(A.template slice<0>(0)));
    CBLAS_TRANSPOSE mat_op_B = internal::to_blas(internal::get_matrix_op(B.template slice<0>(0)));
    CBLAS_INT lda = 0;
    CBLAS_INT ldb = 0;
    CBLAS_INT ldc = C.stride(-2);

    if (mat_op_A == CblasNoTrans) {
      lda = A.stride(-2);
    } else {
      lda = A.stride(-1);
    }

    if (mat_op_B == CblasNoTrans) {
      ldb = B.stride(-2);
    } else {
      ldb = B.stride(-1);
    }

    CBLAS_INT batch_size = static_cast<CBLAS_INT>(C.size(0));
    CBLAS_INT m = C.size(-2);
    CBLAS_INT n = C.size(-1);
    CBLAS_INT k = A.size(-1);
    const Scalar *mat_a = A.data(), *mat_b = B.data();
    Scalar* mat_c = C.data();

    if constexpr (std::is_same_v<float, Scalar>) {
      cblas_sgemm_batch_strided(
        CblasRowMajor,
        mat_op_A, mat_op_B,
        m, n, k,
        alpha, mat_a, lda, A.stride(0),
        mat_b, ldb, B.stride(0),
        beta,
        mat_c, ldc, C.stride(0),
        batch_size
      );
    } else if constexpr (std::is_same_v<double, Scalar>) {
      cblas_dgemm_batch_strided(
        CblasRowMajor,
        mat_op_A, mat_op_B,
        m, n, k,
        alpha, mat_a, lda, A.stride(0),
        mat_b, ldb, B.stride(0),
        beta,
        mat_c, ldc, C.stride(0),
        batch_size
      );
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported type for BLAS gemm.");
    }

#else
    // Use a iteration + gemm for OpenBLAS and Apple.
    index_t batch_size = C.size(0);
    for (index_t i = 0; i < batch_size; ++i) {
      gemm_impl(alpha, A.slice(i), B.slice(i), beta, C.slice(i));
    }
#endif
  }
};

}  // namespace mathprim::blas
