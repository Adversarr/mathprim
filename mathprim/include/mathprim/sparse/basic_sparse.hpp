#pragma once
#include "mathprim/core/buffer.hpp"

namespace mathprim::sparse {

// Now, we support these storage formats
enum class sparse_format_t { csr, csc, coo };

template <typename T, device_t dev> class basic_sparse_storage {
public:
  explicit basic_sparse_storage(index_t rows, index_t cols,
                                index_t nnz) noexcept
      : rows_(rows), cols_(cols), nnz_(nnz) {
    values_ = std::make_unique<basic_buffer<T, 1, dev>>(nnz);
    outer_ptrs_ = std::make_unique<basic_buffer<index_t, 1, dev>>(rows + 1);
    inner_indices_ = std::make_unique<basic_buffer<index_t, 1, dev>>(nnz);
  }

  index_t rows() const { return rows_; }
  index_t cols() const { return cols_; }
  index_t nnz() const { return nnz_; }

  basic_buffer_ptr<T, 1, dev> values() { return values_; }
  basic_buffer_ptr<index_t, 1, dev> outer_ptrs() { return outer_ptrs_; }
  basic_buffer_ptr<index_t, 1, dev> inner_indices() { return inner_indices_; }

  basic_buffer_ptr<const T, 1, dev> values() const { return values_; }

  basic_buffer_ptr<const index_t, 1, dev> outer_ptrs() const {
    return outer_ptrs_;
  }
  basic_buffer_ptr<const index_t, 1, dev> inner_indices() const {
    return inner_indices_;
  }

private:
  index_t rows_;
  index_t cols_;
  index_t nnz_;

  basic_buffer_ptr<T, 1, dev> values_;
  basic_buffer_ptr<index_t, 1, dev> outer_ptrs_;
  basic_buffer_ptr<index_t, 1, dev> inner_indices_;
};

class csr_format;
class csc_format;
class coo_format; // NOTE: we allow coo format to have duplicated entries

template <typename T, typename storage> class basic_sparse_view;

} // namespace mathprim::sparse
