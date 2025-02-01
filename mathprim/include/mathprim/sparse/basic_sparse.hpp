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
  symmetric,    // currently, we do not support symmetric uplo compression
  hermitian,
  skew_symmetric
};

template <typename Scalar, device_t dev, sparse_format format> class basic_sparse_view {
public:

private:
  const basic_view<Scalar, 1, dev> values_;
  const basic_view<index_t, 1, dev> outer_ptrs_;
  const basic_view<index_t, 1, dev> inner_indices_;

  index_t rows_;
  index_t cols_;
  index_t nnz_;
  sparse_property property_;
};

template <typename Scalar, device_t dev> class basic_sparse_storage {
public:
  explicit basic_sparse_storage(index_t rows, index_t cols, index_t nnz) noexcept :
      rows_(rows), cols_(cols), nnz_(nnz) {}

  index_t rows() const noexcept {
    return rows_;
  }
  index_t cols() const noexcept {
    return cols_;
  }
  index_t nnz() const noexcept {
    return nnz_;
  }

private:
  basic_buffer<Scalar, 1, dev> values_;
  basic_buffer<index_t, 1, dev> outer_ptrs_;
  basic_buffer<index_t, 1, dev> inner_indices_;

  index_t rows_;
  index_t cols_;
  index_t nnz_;
};

}  // namespace mathprim::sparse
