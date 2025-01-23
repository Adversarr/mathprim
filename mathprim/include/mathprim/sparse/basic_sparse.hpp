#pragma once
#include "mathprim/core/buffer.hpp"

namespace mathprim::sparse {

// Now, we support these storage formats
enum class sparse_format_t { csr, csc, coo };

template <typename Scalar, device_t dev> class basic_sparse_storage {
public:
  explicit basic_sparse_storage(index_t rows, index_t cols,
                                index_t nnz) noexcept
      : rows_(rows), cols_(cols), nnz_(nnz) {}

  index_t rows() const { return rows_; }
  index_t cols() const { return cols_; }
  index_t nnz() const { return nnz_; }

private:
  basic_buffer_ptr<Scalar, 1, dev> values_;
  basic_buffer_ptr<index_t, 1, dev> outer_ptrs_;
  basic_buffer_ptr<index_t, 1, dev> inner_indices_;

  index_t rows_;
  index_t cols_;
  index_t nnz_;
};

} // namespace mathprim::sparse
