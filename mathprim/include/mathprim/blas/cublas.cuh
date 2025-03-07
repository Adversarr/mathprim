#pragma once
#include <cublas_v2.h>

#include <library_types.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"
#include "mathprim/core/utils/singleton.hpp"

#define MATHPRIM_INTERNAL_CUBLAS_CHECK(expr)                                   \
  do {                                                                         \
    auto status = (expr);                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("CUBLAS error at" + std::string(__FILE__) +     \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               std::string(cublasGetStatusName(status)));      \
    }                                                                          \
  } while (0)

#define MATHPRIM_INTERNAL_CUBLAS_CHECK_EXIT(expr)                              \
  do {                                                                         \
    auto status = (expr);                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "CUBLAS error at" << __FILE__ << ":" << __LINE__ << ": "    \
                << cublasGetStatusName(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace mathprim {
namespace singletons {

class cublas_context final
    : public internal::basic_singleton<cublas_context, cublasHandle_t> {
  using base = internal::basic_singleton<cublas_context, cublasHandle_t>;
  friend base;
  void create_impl(cublasHandle_t &handle) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK_EXIT(cublasCreate(&handle));
    MATHPRIM_INTERNAL_CUBLAS_CHECK_EXIT(
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    MATHPRIM_INTERNAL_CUBLAS_CHECK_EXIT(
        cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
  }

  void destroy_impl(cublasHandle_t &handle) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK_EXIT(cublasDestroy(handle));
  }
};

} // namespace singletons

namespace blas {
namespace internal {

cublasHandle_t get_cublas_handle() { return singletons::cublas_context::get(); }

template <typename Scalar>
__global__ void emul_kernel(const Scalar *x, Scalar *y, index_t total,
                            index_t inc_x, index_t inc_y) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    y[i * inc_y] *= x[i * inc_x];
  }
}

} // namespace internal

template <typename T>
struct cublas : public basic_blas<cublas<T>, T, device::cuda> {
  // Level 1
  template <typename sshape, typename sstride>
  using view_type = basic_view<T, sshape, sstride, device::cuda>;
  template <typename sshape, typename sstride>
  using const_type = basic_view<const T, sshape, sstride, device::cuda>;

  using base = basic_blas<cublas<T>, T, device::cuda>;
  friend base;
  using Scalar = T;

protected:
  template <typename sshape_dst, typename sstride_dst, typename sshape_src,
            typename sstride_src>
  void copy_impl(view_type<sshape_dst, sstride_dst> dst,
                 const_type<sshape_src, sstride_src> src) {
    auto *handle = internal::get_cublas_handle();
    auto inc_dst = dst.stride(-1);
    auto inc_src = src.stride(-1);
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasScopy(handle, src.size(), src.data(),
                                                 inc_src, dst.data(), inc_dst));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDcopy(handle, src.size(), src.data(),
                                                 inc_src, dst.data(), inc_dst));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  template <typename sshape, typename sstride>
  void scal_impl(T alpha, view_type<sshape, sstride> src) {
    auto *handle = internal::get_cublas_handle();
    auto inc_src = src.stride(-1);
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasSscal(handle, src.size(), &alpha, src.data(), inc_src));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasDscal(handle, src.size(), &alpha, src.data(), inc_src));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  template <typename sshape_src, typename sstride_src, typename sshape_dst,
            typename sstride_dst>
  void swap_impl(view_type<sshape_src, sstride_src> src,
                 view_type<sshape_dst, sstride_dst> dst) {
    auto *handle = internal::get_cublas_handle();
    auto inc_src = src.stride(-1);
    auto inc_dst = dst.stride(-1);
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSswap(handle, src.size(), src.data(),
                                                 inc_src, dst.data(), inc_dst));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDswap(handle, src.size(), src.data(),
                                                 inc_src, dst.data(), inc_dst));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  void axpy_impl(T alpha, const_type<sshape_x, sstride_x> x,
                 view_type<sshape_y, sstride_y> y) {
    auto *handle = internal::get_cublas_handle();
    auto inc_x = x.stride(-1);
    auto inc_y = y.stride(-1);
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSaxpy(
          handle, x.size(), &alpha, x.data(), inc_x, y.data(), inc_y));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDaxpy(
          handle, x.size(), &alpha, x.data(), inc_x, y.data(), inc_y));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  template <typename sshape_x, typename sstride_x, typename sshape_y,
            typename sstride_y>
  T dot_impl(const_type<sshape_x, sstride_x> x,
             const_type<sshape_y, sstride_y> y) {
    auto *handle = internal::get_cublas_handle();
    auto inc_x = x.stride(-1);
    auto inc_y = y.stride(-1);
    T result;
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSdot(
          handle, x.size(), x.data(), inc_x, y.data(), inc_y, &result));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDdot(
          handle, x.size(), x.data(), inc_x, y.data(), inc_y, &result));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
    return result;
  }

  template <typename sshape, typename sstride>
  T norm_impl(const_type<sshape, sstride> x) {
    auto *handle = internal::get_cublas_handle();
    auto inc_x = x.stride(-1);
    T result;
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasSnrm2(handle, x.size(), x.data(), inc_x, &result));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasDnrm2(handle, x.size(), x.data(), inc_x, &result));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
    return result;
  }

  template <typename sshape, typename sstride>
  T asum_impl(const_type<sshape, sstride> x) {
    auto *handle = internal::get_cublas_handle();
    auto inc_x = x.stride(-1);
    T result;
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasSasum(handle, x.size(), x.data(), inc_x, &result));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasDasum(handle, x.size(), x.data(), inc_x, &result));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
    return result;
  }

  template <typename sshape, typename sstride>
  index_t amax_impl(const_type<sshape, sstride> x) {
    auto *handle = internal::get_cublas_handle();
    auto inc_x = x.stride(-1);
    int result;
    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasIsamax(handle, x.size(), x.data(), inc_x, &result));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(
          cublasIdamax(handle, x.size(), x.data(), inc_x, &result));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
    return result - 1; // cublas uses 1-based indexing
  }

  // Y <- alpha * A * X + beta * Y
  template <typename SshapeX, typename SstrideX, typename SshapeY,
            typename SstrideY>
  MATHPRIM_NOINLINE void emul_impl(const_type<SshapeX, SstrideX> x,
                                   view_type<SshapeY, SstrideY> y) {
    auto total = x.shape(0);
    auto block_size = 256;
    auto grid_size = (total + block_size - 1) / block_size;
    internal::emul_kernel<Scalar><<<grid_size, block_size>>>(
        x.data(), y.data(), total, x.stride(-1), y.stride(-1));
  }

  // Level 2
  // y <- alpha * A * x + beta * y
  template <typename sshape_A, typename sstride_A, typename sshape_x,
            typename sstride_x, typename sshape_y, typename sstride_y>
  void gemv_impl(T alpha, const_type<sshape_A, sstride_A> A,
                 const_type<sshape_x, sstride_x> x, T beta,
                 view_type<sshape_y, sstride_y> y) {
    auto *handle = internal::get_cublas_handle();
    auto inc_x = x.stride(-1);
    auto inc_y = y.stride(-1);
    auto a_op = internal::get_matrix_op(A);
    int lda = 0, cols = 0, rows = 0;
    cublasOperation_t op;

    // Our API is row-major, so we need to transpose the matrix
    if (a_op == internal::matrix_op::none) {
      lda = A.stride(0);
      rows = A.shape(1);
      cols = A.shape(0);
      op = CUBLAS_OP_T;
    } else {
      lda = A.stride(1);
      rows = A.shape(0);
      cols = A.shape(1);
      op = CUBLAS_OP_N;
    }

    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSgemv(handle, op, rows, cols, &alpha,
                                                 A.data(), lda, x.data(), inc_x,
                                                 &beta, y.data(), inc_y));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDgemv(handle, op, rows, cols, &alpha,
                                                 A.data(), lda, x.data(), inc_x,
                                                 &beta, y.data(), inc_y));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  // Level 3
  // C <- alpha * A * B + beta * C
  template <typename sshape_A, typename sstride_A, typename sshape_B,
            typename sstride_B, typename sshape_C, typename sstride_C>
  void gemm_impl(T alpha, const_type<sshape_A, sstride_A> A,
                 const_type<sshape_B, sstride_B> B, T beta,
                 view_type<sshape_C, sstride_C> C) {
    auto *handle = internal::get_cublas_handle();
    auto a_op = internal::get_matrix_op(A);
    auto b_op = internal::get_matrix_op(B);
    auto c_op = internal::get_matrix_op(C);
    int lda = a_op == internal::matrix_op::none ? A.stride(0) : A.stride(1);
    int ldb = b_op == internal::matrix_op::none ? B.stride(0) : B.stride(1);
    int ldc = c_op == internal::matrix_op::none ? C.stride(0) : C.stride(1);

    // If c_op == none, then C is row-major, do C.T <- alpha * B.T * A.T + beta
    // * C.T
    int m = 0, n = 0, k = 0;
    k = a_op == internal::matrix_op::none ? A.shape(1) : A.shape(0);
    m = C.shape(0);
    n = C.shape(1);


    // The actual call to cublas.
    cublasOperation_t transA, transB;
    int mm, nn, kk;
    const Scalar* aa, *bb;
    int ldaa, ldbb;
    Scalar* cc = C.data();
    if (c_op == internal::matrix_op::none) {
      mm = n;
      nn = m;
      kk = k;
      transA = b_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      transB = a_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      ldaa = ldb;
      ldbb = lda;
      aa = B.data();
      bb = A.data();
    } else {
      mm = m;
      nn = n;
      kk = k;
      transA = a_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      transB = b_op == internal::matrix_op::none ? CUBLAS_OP_T : CUBLAS_OP_N;
      ldaa = lda;
      ldbb = ldb;
      aa = A.data();
      bb = B.data();
    }

    if constexpr (std::is_same_v<T, float>) {
      // MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSgemm(
      //   handle,
      //   transA, transB,
      //   mm, nn, kk,
      //   &alpha,
      //   aa, ldaa,
      //   bb, ldbb,
      //   &beta, cc, ldc));
      cublasGemmAlgo_t algo = gemm_f32_compute_type_ == CUBLAS_COMPUTE_32F ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasGemmEx(
          handle,
          transA, transB,
          mm, nn, kk,
          &alpha,
          aa, CUDA_R_32F, ldaa,
          bb, CUDA_R_32F, ldbb,
          &beta,
          cc, CUDA_R_32F, ldc,
          gemm_f32_compute_type_, algo));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDgemm(
        handle,
        transA, transB,
        mm, nn, kk,
        &alpha,
        aa, ldaa,
        bb, ldbb,
        &beta, cc, ldc));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  template <typename SshapeA, typename SstrideA, typename SshapeB,
            typename SstrideB, typename SshapeC, typename SstrideC>
  MATHPRIM_NOINLINE void
  gemm_batch_strided_impl(Scalar alpha, const_type<SshapeA, SstrideA> A,
                    const_type<SshapeB, SstrideB> B, Scalar beta,
                    view_type<SshapeC, SstrideC> C) {
    auto *handle = internal::get_cublas_handle();
    auto a_op = internal::get_matrix_op(A.slice(0));
    auto b_op = internal::get_matrix_op(B.slice(0));
    auto c_op = internal::get_matrix_op(C.slice(0));
    int lda = a_op == internal::matrix_op::none ? A.stride(1) : A.stride(2);
    int ldb = b_op == internal::matrix_op::none ? B.stride(1) : B.stride(2);
    int ldc = c_op == internal::matrix_op::none ? C.stride(1) : C.stride(2);

    // If c_op == none, then C is row-major, do C.T <- alpha * B.T * A.T + beta
    // * C.T
    int m = 0, n = 0, k = 0;
    k = a_op == internal::matrix_op::none ? A.shape(2) : A.shape(1);
    m = C.shape(1);
    n = C.shape(2);

    // The actual call to cublas.
    cublasOperation_t transA, transB;
    int mm, nn, kk;
    const Scalar* aa, *bb;
    int ldaa, ldbb;
    Scalar* cc = C.data();
    int stride_aa, stride_bb, stride_c = C.stride(0);
    if (c_op == internal::matrix_op::none) {
      mm = n;
      nn = m;
      kk = k;
      transA = b_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      transB = a_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      ldaa = ldb;
      ldbb = lda;
      aa = B.data();
      bb = A.data();
      stride_aa = B.stride(0);
      stride_bb = A.stride(0);
    } else {
      mm = m;
      nn = n;
      kk = k;
      transA = a_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      transB = b_op == internal::matrix_op::none ? CUBLAS_OP_T : CUBLAS_OP_N;
      ldaa = lda;
      ldbb = ldb;
      aa = A.data();
      bb = B.data();
      stride_aa = A.stride(0);
      stride_bb = B.stride(0);
    }
    index_t batch_size = C.shape(0);

    if constexpr (std::is_same_v<T, float>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        transA, transB,
        mm, nn, kk,
        &alpha,
        aa, ldaa, stride_aa,
        bb, ldbb, stride_bb,
        &beta, cc, ldc, stride_c,
        batch_size));
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDgemmStridedBatched(
        handle,
        transA, transB,
        mm, nn, kk,
        &alpha,
        aa, ldaa, stride_aa,
        bb, ldbb, stride_bb,
        &beta, cc, ldc, stride_c,
        batch_size));
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }



  template <typename SshapeX, typename SstrideX, typename SshapeY, typename SstrideY>
  void axpby_impl(Scalar alpha, const_type<SshapeX, SstrideX> x, Scalar beta, view_type<SshapeY, SstrideY> y) {
    // No native support
    scal_impl(beta, y);
    axpy_impl(alpha, x, y);
  }

public:
  cublasComputeType_t gemm_f32_compute_type_ = CUBLAS_COMPUTE_32F;
  cublasComputeType_t gemm_f64_compute_type_ = CUBLAS_COMPUTE_64F;
};

} // namespace blas
#undef MATHPRIM_INTERNAL_CUBLAS_CHECK

} // namespace mathprim
