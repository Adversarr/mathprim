#pragma once

#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse {
namespace blas {
template <typename Scalar, sparse_format SparseCompression, typename Backend = par::seq>
class naive;

template <typename Scalar, typename Backend>
class naive<Scalar, sparse_format::csr, Backend>
    : public sparse_blas_base<naive<Scalar, sparse_format::csr, Backend>, Scalar, device::cpu, sparse_format::csr> {
public:
  using base = sparse_blas_base<naive<Scalar, sparse_format::csr, Backend>, Scalar, device::cpu, sparse_format::csr>;
  friend base;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  using base::base;

private:
  template <typename SshapeB, typename SstrideB, typename SshapeC, typename SstrideC>
  void spmm_impl(Scalar alpha, basic_view<const Scalar, SshapeB, SstrideB, device::cpu> B, Scalar beta,
                 basic_view<Scalar, SshapeC, SstrideC, device::cpu> C, bool transA = false) {
    for (index_t i = 0; i < C.shape(0); ++i) {
      for (index_t j = 0; j < C.shape(1); ++j) {
        C(i, j) *= beta;
      }
    }

    if (transA) {
      visit(this->mat_, par::seq{}, [&](index_t i, index_t k, Scalar value) {
        for (index_t j = 0; j < C.shape(1); ++j) {
          C(k, j) += alpha * value * B(i, j);
        }
      });
    } else {
      visit(this->mat_, par::seq{}, [&](index_t i, index_t k, Scalar value) {
        for (index_t j = 0; j < C.shape(1); ++j) {
          C(i, j) += alpha * value * B(k, j);
        }
      });
    }
  }

  // y = alpha * A * x + beta * y.
  void gemv_impl(Scalar alpha, const_vector_view x, Scalar beta, vector_view y, bool transpose) {
    if (transpose) {  // Computes A.T @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        gemv_no_trans(alpha, x, beta, y);
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        gemv_no_trans(-alpha, x, beta, y);
      } else {
        gemv_trans(alpha, x, beta, y);  // always slower sequential
      }
    } else {  // Computes A @ x
      gemv_no_trans(alpha, x, beta, y);
    }
  }

  void gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  void gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
};

template <typename Scalar, typename Backend>
class naive<Scalar, sparse_format::csc, Backend>
    : public sparse_blas_base<naive<Scalar, sparse_format::csc, Backend>, Scalar, device::cpu, sparse_format::csc> {
public:
  using base = sparse_blas_base<naive<Scalar, sparse_format::csc, Backend>, Scalar, device::cpu, sparse_format::csc>;
  friend base;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using base::base;

private:
  template <typename SshapeB, typename SstrideB, typename SshapeC, typename SstrideC>
  void spmm_impl(Scalar alpha, basic_view<Scalar, SshapeB, SstrideB, device::cpu> B, Scalar beta,
                 basic_view<Scalar, SshapeC, SstrideC, device::cpu> C, bool transA = false) {
    for (index_t i = 0; i < C.shape(0); ++i) {
      for (index_t j = 0; j < C.shape(1); ++j) {
        C(i, j) *= beta;
      }
    }

    if (transA) {
      visit(this->mat_, par::seq{}, [&](index_t i, index_t k, Scalar value) {
        for (index_t j = 0; j < C.shape(1); ++j) {
          C(k, j) += alpha * value * B(i, j);
        }
      });
    } else {
      visit(this->mat_, par::seq{}, [&](index_t i, index_t k, Scalar value) {
        for (index_t j = 0; j < C.shape(1); ++j) {
          C(i, j) += alpha * value * B(k, j);
        }
      });
    }
  }

  // y = alpha * A * x + beta * y.
  void gemv_impl(Scalar alpha, const_vector_view x, Scalar beta, vector_view y, bool transpose) {
    if (transpose) {  // Computes A.T @ x
      gemv_trans(alpha, x, beta, y);
    } else {  // Computes A @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        gemv_trans(alpha, x, beta, y);
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        gemv_trans(-alpha, x, beta, y);
      } else {
        gemv_no_trans(alpha, x, beta, y);
      }
    }
  }

  void gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  void gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
};

///////////////////////////////////////////////////////////////////////////////
/// Implementation for CSR format.
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar, typename Backend>
void naive<Scalar, sparse_format::csr, Backend>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                               vector_view y) {
  const auto& mat = this->mat_;
  const auto row_ptr = mat.outer_ptrs();
  const auto col_ind = mat.inner_indices();
  const auto values = mat.values();
  const auto m = mat.rows();

  Backend parfor;
  parfor.run(make_shape(m), [row_ptr, col_ind, values, x, y, alpha, beta](index_t i) {
    Scalar result = 0;
    const index_t beg = row_ptr[i], end = row_ptr[i + 1];

    for (index_t j = beg; j < end; ++j) {
      result += values[j] * x[col_ind[j]];
    }
    y[i] = alpha * result + beta * y[i];
  });
}

template <typename Scalar, typename Backend>
void naive<Scalar, sparse_format::csr, Backend>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                            vector_view y) {
  const auto& mat = this->mat_;
  const auto& row_ptr = mat.outer_ptrs();
  const auto& col_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto m = mat.rows();
  const auto n = mat.cols();
  // Initialize y with beta * y
  MATHPRIM_PRAGMA_UNROLL_HOST
  for (index_t i = 0; i < n; ++i) {
    y[i] *= beta;
  }

  for (index_t i = 0; i < m; ++i) {
    index_t beg = row_ptr[i], end = row_ptr[i + 1];
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t j = beg; j < end; ++j) {
      y[col_ind[j]] += alpha * values[j] * x[i];
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
/// Implementation for CSC format.
///////////////////////////////////////////////////////////////////////////////
template <typename Scalar, typename Backend>
void naive<Scalar, sparse_format::csc, Backend>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                               vector_view y) {
  const auto& mat = this->mat_;
  const auto& col_ptr = mat.outer_ptrs();
  const auto& row_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto m = mat.rows();
  const auto n = mat.cols();

  // Initialize y with beta * y
  MATHPRIM_PRAGMA_UNROLL_HOST
  for (index_t i = 0; i < m; ++i) {
    y[i] *= beta;
  }

  for (index_t j = 0; j < n; ++j) {
    const index_t beg = col_ptr[j], end = col_ptr[j + 1];

    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = beg; i < end; ++i) {
      y[row_ind[i]] += alpha * values[i] * x[j];
    }
  }
}

template <typename Scalar, typename Backend>
void naive<Scalar, sparse_format::csc, Backend>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta,
                                                            vector_view y) {
  const auto& mat = this->mat_;
  const auto& col_ptr = mat.outer_ptrs();
  const auto& row_ind = mat.inner_indices();
  const auto& values = mat.values();
  const auto n = mat.cols();

  Backend parfor;
  parfor.run(make_shape(n), [col_ptr, row_ind, values, x, y, alpha, beta](index_t j) {
    index_t beg = col_ptr[j], end = col_ptr[j + 1];
    Scalar result = 0;
    for (index_t i = beg; i < end; ++i) {
      result += values[i] * x[row_ind[i]];
    }
    y[j] = alpha * result + beta * y[j];
  });
}

}  // namespace blas
}  // namespace mathprim::sparse
