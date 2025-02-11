#pragma once

#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
#include <algorithm>

namespace mathprim::sparse {
namespace blas {
template <typename Scalar, typename device, sparse_format sparse_compression, typename backend = par::seq>
class naive;

template <typename Scalar, typename backend>
class naive<Scalar, device::cpu, sparse_format::csr, backend>
    : public sparse_blas_base<Scalar, device::cpu, sparse_format::csr> {
public:
  using base = sparse_blas_base<Scalar, device::cpu, sparse_format::csr>;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using base::base;

  // y = alpha * A * x + beta * y.
  void gemv(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) override {
    this->check_gemv_shape(x, y);
    if (this->mat_.is_transpose()) { // Computes A.T @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        gemv_no_trans(alpha, x, beta, y);
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        gemv_no_trans(-alpha, x, beta, y);
      }

      gemv_trans(alpha, x, beta, y); // always slower sequential
    } else { // Computes A @ x
      gemv_no_trans(alpha, x, beta, y);
    }
  }

private:
  void gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  void gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
};

///////////////////////////////////////////////////////////////////////////////
/// Implementation for the naive CPU version
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar, typename backend>
void naive<Scalar, device::cpu, sparse_format::csr, backend>::gemv_no_trans(Scalar alpha, const_vector_view x,
                                                                            Scalar beta, vector_view y) {
  const auto& mat = this->mat_;
  const auto row_ptr = mat.outer_ptrs();
  const auto col_ind = mat.inner_indices();
  const auto values = mat.values();
  const auto m = mat.rows();
  const auto n = mat.cols();
  const auto nnz = mat.nnz();

  backend parfor;
  parfor.run(make_shape(m), [row_ptr, col_ind, values, x, y, alpha, beta](index_t i) {
    Scalar result = 0;
    for (index_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      result += values[j] * x[col_ind[j]];
    }
    y[i] = alpha * result + beta * y[i];
  });
}

template <typename Scalar, typename backend>
void naive<Scalar, device::cpu, sparse_format::csr, backend>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                                         vector_view y) {
  const auto& mat = this->mat_;
  const auto& row_ptr = mat.outer_ptrs();
  const auto& col_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto m = mat.rows();
  const auto n = mat.cols();
  const auto nnz = mat.nnz();

  // Perform the matrix-vector multiplication for the transposed matrix
  // if constexpr (std::is_same_v<backend, par::seq>) {
  // Initialize y with beta * y
  for (index_t i = 0; i < n; ++i) {
    y[i] *= beta;
  }
  for (index_t i = 0; i < m; ++i) {
    for (index_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      y[col_ind[j]] += alpha * values[j] * x[i];
    }
  }
  // } else { // Super Slow, do not use
  //   backend par;
  //   par.run(make_shape(m), [row_ptr, col_ind, values, x, y, alpha, beta, m](index_t j) {
  //     Scalar result = 0;  // y[j] = sum_i A[i, j] * x[i]
  //     for (index_t i = 0; i < m; ++i) {
  //       // find j
  //       index_t beg = row_ptr[i], end = row_ptr[i + 1];

  //       if (j < col_ind[beg] || j >= col_ind[end - 1]) {
  //         continue;
  //       } else if (j == col_ind[beg]) {
  //         result += values[beg] * x[i];
  //         continue;
  //       } else if (j == col_ind[end - 1]) {
  //         result += values[end - 1] * x[i];
  //         continue;
  //       }

  //       // binary search
  //       index_t l = beg, r = end - 1;
  //       while (l < r) {
  //         index_t mid = (l + r) / 2;
  //         if (col_ind[mid] == j) {
  //           result += values[mid] * x[i];
  //           break;
  //         } else if (col_ind[mid] < j) {
  //           l = mid + 1;
  //         } else {
  //           r = mid;
  //         }
  //       }
  //     }
  //     y[j] = alpha * result + beta * y[j];
  //   });
  // }
}

}  // namespace blas
}  // namespace mathprim::sparse
