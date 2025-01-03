#pragma once

#include "mathprim/core/defines.hpp"

namespace mathprim::blas {

template <typename T, typename blas_t, device_t dev>
struct backend_blas {
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

}  // namespace mathprim::blas
