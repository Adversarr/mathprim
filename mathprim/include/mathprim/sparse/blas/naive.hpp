#pragma once

#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse {
namespace blas {
template <typename Scalar, sparse_format sparse_compression, typename backend = par::seq>
class naive;

template <typename Scalar, typename backend>
class naive<Scalar, sparse_format::csr, backend> : public sparse_blas_base<Scalar, device::cpu, sparse_format::csr> {
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
    if (this->mat_.is_transpose()) {  // Computes A.T @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        gemv_no_trans(alpha, x, beta, y);
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        gemv_no_trans(-alpha, x, beta, y);
      }

      gemv_trans(alpha, x, beta, y);  // always slower sequential
    } else {                          // Computes A @ x
      gemv_no_trans(alpha, x, beta, y);
    }
  }

private:
  void gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  void gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
};

template <typename Scalar, typename backend>
class naive<Scalar, sparse_format::csc, backend> : public sparse_blas_base<Scalar, device::cpu, sparse_format::csc> {
public:
  using base = sparse_blas_base<Scalar, device::cpu, sparse_format::csc>;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using base::base;

  // y = alpha * A * x + beta * y.
  void gemv(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) override {
    this->check_gemv_shape(x, y);
    if (this->mat_.is_transpose()) {  // Computes A.T @ x
      gemv_trans(alpha, x, beta, y);
    } else {  // Computes A @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        gemv_trans(alpha, x, beta, y);
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        gemv_trans(-alpha, x, beta, y);
      }
      gemv_no_trans(alpha, x, beta, y);
    }
  }

private:
  void gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  void gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
};

///////////////////////////////////////////////////////////////////////////////
/// Implementation for CSR format.
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar, typename backend>
void naive<Scalar, sparse_format::csr, backend>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                               vector_view y) {
  const auto& mat = this->mat_;
  const auto row_ptr = mat.outer_ptrs();
  const auto col_ind = mat.inner_indices();
  const auto values = mat.values();
  const auto m = mat.rows();

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
void naive<Scalar, sparse_format::csr, backend>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                            vector_view y) {
  const auto& mat = this->mat_;
  const auto& row_ptr = mat.outer_ptrs();
  const auto& col_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto m = mat.rows();
  const auto n = mat.cols();
  // Initialize y with beta * y
  for (index_t i = 0; i < n; ++i) {
    y[i] *= beta;
  }

  for (index_t i = 0; i < m; ++i) {
    for (index_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      y[col_ind[j]] += alpha * values[j] * x[i];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Implementation for CSC format.
///////////////////////////////////////////////////////////////////////////////
template <typename Scalar, typename backend>
void naive<Scalar, sparse_format::csc, backend>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                               vector_view y) {
  const auto& mat = this->mat_;
  const auto& col_ptr = mat.outer_ptrs();
  const auto& row_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto m = mat.rows();
  const auto n = mat.cols();

  // Initialize y with beta * y
  for (index_t i = 0; i < m; ++i) {
    y[i] *= beta;
  }

  for (index_t j = 0; j < n; ++j) {
    for (index_t i = col_ptr[j]; i < col_ptr[j + 1]; ++i) {
      y[row_ind[i]] += alpha * values[i] * x[j];
    }
  }
}

template <typename Scalar, typename backend>
void naive<Scalar, sparse_format::csc, backend>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                            vector_view y) {
  const auto& mat = this->mat_;
  const auto& col_ptr = mat.outer_ptrs();
  const auto& row_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto n = mat.cols();

  backend parfor;
  parfor.run(make_shape(n), [col_ptr, row_ind, values, x, y, alpha, beta](index_t j) {
    for (index_t i = col_ptr[j]; i < col_ptr[j + 1]; ++i) {
      y[j] += alpha * values[i] * x[row_ind[i]];
    }
  });
}

}  // namespace blas
}  // namespace mathprim::sparse
