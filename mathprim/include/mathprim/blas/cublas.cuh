#pragma once
#include <cublas_v2.h>

#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"

namespace mathprim {

namespace blas {

namespace internal {

inline void check_status(cublasStatus_t status, const char *file, int line,
                         const char *expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "CUBLAS error at " << file << ":" << line << ": " << expr << " "
        << cublasGetStatusName(status);
    throw std::runtime_error(oss.str());
  }
}

#define MATHPRIM_INTERNAL_CUBLAS_CHECK(expr)                                   \
  ::mathprim::blas::internal::check_status((expr), __FILE__, __LINE__, #expr)

class cublas_context final {
public:
  cublas_context() {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasCreate(&handle_));
    cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_ALLOWED);
  }
  ~cublas_context() { MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDestroy(handle_)); }
  cublas_context(const cublas_context &) = delete;
  cublas_context &operator=(const cublas_context &) = delete;
  cublas_context(cublas_context &&) = delete;
  cublas_context &operator=(cublas_context &&) = delete;

  cublasHandle_t handle() const { return handle_; }

  static cublas_context &instance() {
    static cublas_context ctx;
    return ctx;
  }

private:
  cublasHandle_t handle_;
};

cublasHandle_t get_cublas_handle() {
  return cublas_context::instance().handle();
}

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
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSgemm(
        handle,
        transA, transB,
        mm, nn, kk,
        &alpha,
        aa, ldaa,
        bb, ldbb,
        &beta, cc, ldc));
      // if (c_op == internal::matrix_op::none) { // row major.
      //   cublasOperation_t opA =
      //       a_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      //   cublasOperation_t opB =
      //       b_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      //   // transpose A, B, C, and do: C.T <- alpha op(B.T) op(A.T) + beta C.T
      //   MATHPRIM_INTERNAL_CUBLAS_CHECK(
      //       cublasSgemm(handle, opB, opA, n, m, k, &alpha, B.data(), ldb,
      //                   A.data(), lda, &beta, C.data(), ldc));
      // } else { // C is column major, i.e. C is a transpose view.
      //   cublasOperation_t opA =
      //       a_op == internal::matrix_op::none ? CUBLAS_OP_T : CUBLAS_OP_N;
      //   cublasOperation_t opB =
      //       b_op == internal::matrix_op::none ? CUBLAS_OP_T : CUBLAS_OP_N;
      //   // C.T <- alpha * op(A) * op(B) + beta * C.T
      //   MATHPRIM_INTERNAL_CUBLAS_CHECK(
      //       cublasSgemm(handle, opA, opB, m, n, k, &alpha, A.data(), lda,
      //                   B.data(), ldb, &beta, C.data(), ldc));
      // }
    } else if constexpr (std::is_same_v<T, double>) {
      MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDgemm(
        handle,
        transA, transB,
        mm, nn, kk,
        &alpha,
        aa, ldaa,
        bb, ldbb,
        &beta, cc, ldc));
      // if (c_op == internal::matrix_op::none) { // row major.
      //   cublasOperation_t opA =
      //       a_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      //   cublasOperation_t opB =
      //       b_op == internal::matrix_op::none ? CUBLAS_OP_N : CUBLAS_OP_T;
      //   // transpose A, B, C, and do: C.T <- alpha op(B.T) op(A.T) + beta C.T
      //   MATHPRIM_INTERNAL_CUBLAS_CHECK(
      //       cublasDgemm(handle, opB, opA, n, m, k, &alpha, B.data(), ldb,
      //                   A.data(), lda, &beta, C.data(), ldc));
      // } else { // C is column major, i.e. C is a transpose view.
      //   cublasOperation_t opA =
      //       a_op == internal::matrix_op::none ? CUBLAS_OP_T : CUBLAS_OP_N;
      //   cublasOperation_t opB =
      //       b_op == internal::matrix_op::none ? CUBLAS_OP_T : CUBLAS_OP_N;
      //   // C.T <- alpha * op(A) * op(B) + beta * C.T
      //   MATHPRIM_INTERNAL_CUBLAS_CHECK(
      //       cublasDgemm(handle, opA, opB, m, n, k, &alpha, A.data(), lda,
      //                   B.data(), ldb, &beta, C.data(), ldc));
      // }
    } else {
      static_assert(::mathprim::internal::always_false_v<T>,
                    "Unsupported type");
    }
  }

  template <typename SshapeA, typename SstrideA, typename SshapeB,
            typename SstrideB, typename SshapeC, typename SstrideC>
  MATHPRIM_NOINLINE void
  gemm_batched_impl(Scalar alpha, const_type<SshapeA, SstrideA> A,
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
};

} // namespace blas
#undef MATHPRIM_INTERNAL_CUBLAS_CHECK

} // namespace mathprim
