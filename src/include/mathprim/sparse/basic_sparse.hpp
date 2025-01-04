#pragma once
#include "mathprim/core/buffer.hpp"

namespace mathprim {

template <typename T, device_t dev> class basic_sparse_matrix {
public:
  basic_sparse_matrix(index_t rows, index_t cols, index_t nnz) :
      rows_(rows), cols_(cols), nnz_(nnz) {
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

  void set_values(basic_buffer_ptr<T, 1, dev> values) {
    values_ = std::move(values);
  }
  void set_outer_ptrs(basic_buffer_ptr<index_t, 1, dev> outer_ptrs) {
    outer_ptrs_ = std::move(outer_ptrs);
  }
  void set_inner_indices(basic_buffer_ptr<index_t, 1, dev> inner_indices) {
    inner_indices_ = std::move(inner_indices);
  }

  void set_values(basic_buffer_ptr<const T, 1, dev> values) {
    values_ = std::make_unique<basic_buffer<T, 1, dev>>(values->size());
    values_->copy_from(*values);
  }
  void set_outer_ptrs(basic_buffer_ptr<const index_t, 1, dev> outer_ptrs) {
    outer_ptrs_
        = std::make_unique<basic_buffer<index_t, 1, dev>>(outer_ptrs->size());
    outer_ptrs_->copy_from(*outer_ptrs);
  }
  void set_inner_indices(
      basic_buffer_ptr<const index_t, 1, dev> inner_indices) {
    inner_indices_ = std::make_unique<basic_buffer<index_t, 1, dev>>(
        inner_indices->size());
    inner_indices_->copy_from(*inner_indices);
  }

private:
  index_t rows_;
  index_t cols_;
  index_t nnz_;

  basic_buffer_ptr<T, 1, dev> values_;
  basic_buffer_ptr<index_t, 1, dev> outer_ptrs_;
  basic_buffer_ptr<index_t, 1, dev> inner_indices_;
};

}  // namespace mathprim
