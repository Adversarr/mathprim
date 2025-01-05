#pragma once
#include "mathprim/core/utils/cuda_utils.cuh"

#include <cublas_v2.h>
#include <sstream>

#include "mathprim/core/defines.hpp"

namespace mathprim {

namespace blas {

namespace internal {

class cublas_context final {
public:
  cublas_context() { cublasCreate(&handle_); }
  ~cublas_context() { cublasDestroy(handle_); }
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

void check_status(cublasStatus_t status, const char *file, int line,
                  const char *expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "CUBLAS error at " << file << ":" << line << ": " << expr << " "
        << cublasGetStatusName(status);
    throw std::runtime_error(oss.str());
  }
}

#define MATHPRIM_INTERNAL_CUBLAS_CHECK(expr)                                   \
  mathprim::blas::internal::check_status((expr), __FILE__, __LINE__, #expr)

} // namespace internal
template <typename T> struct blas_impl_cublas {
  static constexpr device_t dev = device_t::cuda;
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

} // namespace blas

} // namespace mathprim

namespace mathprim::blas {

template <typename T>
void blas_impl_cublas<T>::copy(vector_view dst, const_vector_view src) {
  internal::cublas_context::instance();
  auto &ctx = internal::cublas_context::instance();
  auto *handle = ctx.handle();
  auto n = static_cast<int>(src.size(0));
  auto src_ptr = src.data();
  auto dst_ptr = dst.data();
  auto incx = src.stride(0);
  auto incy = dst.stride(0);
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasScopy(handle, n, src_ptr, incx, dst_ptr, incy));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDcopy(handle, n, src_ptr, incx, dst_ptr, incy));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}

template <typename T> void blas_impl_cublas<T>::scal(T alpha, vector_view x) {
  static_assert(!std::is_const_v<T>, "T must be non-const");
  auto *handle = internal::cublas_context::instance().handle();
  auto n = static_cast<int>(x.size(0));
  T *x_ptr = x.data();
  index_t incx = x.stride(0);
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasSscal_v2(handle, n, &alpha, x_ptr, incx));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDscal_v2(handle, n, &alpha, x_ptr, incx));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}
template <typename T>
void blas_impl_cublas<T>::swap(vector_view x, vector_view y) {
  internal::cublas_context::instance();
  auto &ctx = internal::cublas_context::instance();
  auto *handle = ctx.handle();
  auto n = static_cast<int>(x.size(0));
  auto x_ptr = x.data();
  auto y_ptr = y.data();
  auto incx = x.stride(0);
  auto incy = y.stride(0);
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasSswap(handle, n, x_ptr, incx, y_ptr, incy));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDswap(handle, n, x_ptr, incx, y_ptr, incy));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}

template <typename T>
void blas_impl_cublas<T>::axpy(T alpha, const_vector_view x, vector_view y) {
  internal::cublas_context::instance();
  auto &ctx = internal::cublas_context::instance();
  auto *handle = ctx.handle();
  auto n = static_cast<int>(y.size(0));
  auto x_ptr = x.data();
  auto y_ptr = y.data();
  auto incx = x.stride(0);
  auto incy = y.stride(0);
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasSaxpy(handle, n, &alpha, x_ptr, incx, y_ptr, incy));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDaxpy(handle, n, &alpha, x_ptr, incx, y_ptr, incy));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}

template <typename T>
T blas_impl_cublas<T>::dot(const_vector_view x, const_vector_view y) {
  internal::cublas_context::instance();
  auto &ctx = internal::cublas_context::instance();
  auto *handle = ctx.handle();
  auto n = static_cast<int>(x.size(0));
  auto x_ptr = x.data();
  auto y_ptr = y.data();
  auto incx = x.stride(0);
  auto incy = y.stride(0);
  T result;
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasSdot(handle, n, x_ptr, incx, y_ptr, incy, &result));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDdot(handle, n, x_ptr, incx, y_ptr, incy, &result));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
  return result;
}

template <typename T> T blas_impl_cublas<T>::norm(const_vector_view x) {
  internal::cublas_context::instance();
  auto &ctx = internal::cublas_context::instance();
  auto *handle = ctx.handle();
  auto n = static_cast<int>(x.size(0));
  auto x_ptr = x.data();
  auto incx = x.stride(0);
  T result;
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasSnrm2(handle, n, x_ptr, incx, &result));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDnrm2(handle, n, x_ptr, incx, &result));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
  return result;
}

template <typename T> T blas_impl_cublas<T>::asum(const_vector_view x) {
  internal::cublas_context::instance();
  auto &ctx = internal::cublas_context::instance();
  auto *handle = ctx.handle();
  auto n = static_cast<int>(x.size(0));
  auto x_ptr = x.data();
  auto incx = x.stride(0);
  T result;
  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasSasum(handle, n, x_ptr, incx, &result));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(
        cublasDasum(handle, n, x_ptr, incx, &result));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}

// template <typename T>
// MATHPRIM_BACKEND_BLAS_AMAX_IMPL(T, blas_cuda_cublas, device_t::cuda) {
//   internal::cublas_context::instance();
//   auto &ctx = internal::cublas_context::instance();
//   auto *handle = ctx.handle();
//   auto n = static_cast<int>(x.size());
//   auto x_ptr = x.data();
//   auto incx = x.stride(0);
//   int result;
//   if constexpr (std::is_same_v<T, float>) {
//     MATHPRIM_INTERNAL_CUBLAS_CHECK(
//         cublasIsamax(handle, n, x_ptr, incx, &result));
//   } else if constexpr (std::is_same_v<T, double>) {
//     MATHPRIM_INTERNAL_CUBLAS_CHECK(
//         cublasIdamax(handle, n, x_ptr, incx, &result));
//   } else {
//     static_assert(!std::is_same_v<T, T>, "Unsupported type");
//   }
//   return result - 1; // cublas returns 1-based index
// }

template <typename T>
void blas_impl_cublas<T>::gemv(T alpha, const_matrix_view A,
                               const_vector_view x, T beta, vector_view y) {
  int m = static_cast<int>(A.size(0));
  int n = static_cast<int>(A.size(1));
  int row_stride = static_cast<int>(A.stride(0));
  int col_stride = static_cast<int>(A.stride(1));
  int inc_x = static_cast<int>(x.stride(0));
  int inc_y = static_cast<int>(y.stride(0));
  auto *handle = internal::cublas_context::instance().handle();
  int lda = col_stride;
  cublasOperation_t op = CUBLAS_OP_T;
  if (row_stride == 1 && col_stride > 1) {
    op = CUBLAS_OP_N;
    lda = row_stride;
  } else if (row_stride > 1 && col_stride == 1) {
    lda = row_stride;
    std::swap(m, n);
  } else {
    MATHPRIM_INTERNAL_FATAL("Invalid matrix view for GEMV");
  }

  if constexpr (std::is_same_v<T, float>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasSgemv(handle, op, m, n, &alpha,
                                               A.data(), lda, x.data(), inc_x,
                                               &beta, y.data(), inc_y));
  } else if constexpr (std::is_same_v<T, double>) {
    MATHPRIM_INTERNAL_CUBLAS_CHECK(cublasDgemv(handle, op, m, n, &alpha,
                                               A.data(), lda, x.data(), inc_x,
                                               &beta, y.data(), inc_y));
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}

} // namespace mathprim::blas