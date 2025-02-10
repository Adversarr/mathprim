#pragma once
#include "mathprim/core/buffer.hpp"

namespace mathprim::sparse {

enum class sparse_format {
  csr,  // in Eigen, corresponding to compressed sparse row format
  csc,  // in Eigen, corresponding to compressed sparse column format
  coo,  // in coo format, we assume that the indices are sorted by row
  bsr,  // blocked compress row.
};

enum class sparse_property {
  general,
  symmetric,  // currently, we do not support symmetric uplo compression
  hermitian,
  skew_symmetric
};

template <typename Scalar, typename device>
class basic_sparse_view {
public:
  using values_view = continuous_view<Scalar, shape_t<keep_dim>, device>;
  using ptrs_view = continuous_view<index_t, shape_t<keep_dim>, device>;

private:
  values_view values_;
  ptrs_view outer_ptrs_;
  ptrs_view inner_indices_;

  index_t rows_;
  index_t cols_;
  index_t nnz_;
  sparse_property property_{sparse_property::general};
  bool transpose_{false};
};

// Sparse BLAS basic API.
template <typename Scalar, typename device>
class sparse_blas_base {
public:
  using vector_view = continuous_view<Scalar, shape_t<keep_dim>, device>;
  using const_vector_view = continuous_view<const Scalar, shape_t<keep_dim>, device>;
  using sparse_view = basic_sparse_view<Scalar, device>;
  using const_sparse_view = basic_sparse_view<const Scalar, device>;
  explicit sparse_blas_base(const_sparse_view matrix_view) : mat_(matrix_view) {}

  // y = alpha * A * x + beta * y.
  virtual void gemv(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) = 0;

  // // <x, y>_A = x^T * A * y.
  // virtual Scalar inner(const_vector_view x, const_vector_view y) = 0;

protected:
  const_sparse_view mat_;
};

}  // namespace mathprim::sparse
