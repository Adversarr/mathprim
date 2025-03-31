/**
 * @brief These algorithms are modified from nvidia's cusplibrary. We include their LICENSE here.
 *
 */

/*
 *  Copyright 2008-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once
#include <Eigen/Sparse>
#include <map>
#include <vector>

#include "mathprim/core/defines.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse::iterative {
namespace detail {

template <typename T>
bool less_than_abs(const T &a, const T &b) {
  T abs_a = a < 0 ? -a : a;
  T abs_b = b < 0 ? -b : b;
  return abs_a < abs_b;
}

template <typename ValueType>
class ainv_matrix_row {
public:
  struct map_entry {
    ValueType value;
    int heapidx;

    map_entry(const ValueType &v, int i) : value(v), heapidx(i) {}
    map_entry() {}
  };

  struct heap_entry {
    ValueType value;
    typename std::map<index_t, typename ainv_matrix_row::map_entry>::iterator mapiter;

    heap_entry(const ValueType &v, const typename std::map<index_t, typename ainv_matrix_row::map_entry>::iterator &i) :
        value(v), mapiter(i) {}
    heap_entry() {}
  };

  using const_iterator = typename std::map<index_t, typename ainv_matrix_row::map_entry>::const_iterator;

private:
  typename std::map<index_t, typename ainv_matrix_row::map_entry> row_map_;  // row entries sorted by index
  typename std::vector<typename ainv_matrix_row::heap_entry>
      row_heap_;  // row entries sorted by min-abs-val (in a heap)

  void heap_swap(int i, int j) {
    // swap the entries
    typename ainv_matrix_row::heap_entry val = this->row_heap_[i];
    this->row_heap_[i] = this->row_heap_[j];
    this->row_heap_[j] = val;

    // update the backpointers
    this->row_heap_[i].mapiter->second.heapidx = i;
    this->row_heap_[j].mapiter->second.heapidx = j;
  }

  void downheap(int i) {
    int child0 = (i + 1) * 2 - 1;
    int child1 = (i + 1) * 2;

    while (static_cast<size_t>(child0) < this->row_heap_.size()
           || static_cast<size_t>(child1) < this->row_heap_.size()) {
      int min_child = child0;  // this will be the child with the lowest value that is in-bounds.
      if (static_cast<size_t>(child1) < this->row_heap_.size()
          && less_than_abs(this->row_heap_[child1].value, this->row_heap_[child0].value))
        min_child = child1;
      // if either child is lower, swap with whichever is smaller, otherwise we're done
      if (less_than_abs(this->row_heap_[child0].value, this->row_heap_[i].value)
          || (static_cast<size_t>(child1) < this->row_heap_.size()
              && less_than_abs(this->row_heap_[child1].value, this->row_heap_[i].value)))
        this->heap_swap(i, min_child);
      else
        break;

      i = min_child;
      child0 = (i + 1) * 2 - 1;
      child1 = (i + 1) * 2;
    }
  }

  void upheap(int i) {
    int parent = (i - 1) / 2;
    while (i != 0) {
      if (less_than_abs(this->row_heap_[i].value, this->row_heap_[parent].value))
        this->heap_swap(i, parent);
      else
        break;

      i = parent;
      parent = (i - 1) / 2;
    }
  }

  void heap_insert(typename ainv_matrix_row::heap_entry val) {
    this->row_heap_.push_back(val);
    val.mapiter->second.heapidx = static_cast<int>(this->row_heap_.size()) - 1;
    upheap(this->row_heap_.size() - 1);
  }

  void heap_pop() {
    if (this->row_heap_.empty())
      return;

    heap_swap(0, this->row_heap_.size() - 1);
    // no need to erase the backpointer, since the tree will be updated elsewhere

    this->row_heap_.pop_back();

    downheap(0);
  }

  void heap_update(int i, ValueType val) {
    ValueType old_val = this->row_heap_[i].value;
    this->row_heap_[i].value = val;

    if (less_than_abs(val, old_val))
      upheap(i);
    else
      downheap(i);
  }

public:
  typename ainv_matrix_row::const_iterator begin() const { return this->row_map_.begin(); }
  typename ainv_matrix_row::const_iterator end() const { return this->row_map_.end(); }
  size_t size() const { return this->row_map_.size(); }

  bool has_entry_at_index(index_t i) { return this->row_map_.count(i) != 0; }

  void mult_by_scalar(ValueType scalar) {
    // since we already have a table of pointers into the map, this is O(n) via pointer chasing
    for (int i = 0; static_cast<size_t>(i) < this->row_heap_.size(); i++) {
      this->row_heap_[i].value *= scalar;
      this->row_heap_[i].mapiter->second.value *= scalar;
    }
  }

  void insert(index_t i, ValueType t) {
    ainv_matrix_row::map_entry me(t, -1);
    ainv_matrix_row::heap_entry he;

    // map::insert returns a pair (iterator, bool), so we can grab the iterator from that
    he.mapiter = this->row_map_.insert(std::make_pair(i, me)).first;
    he.value = t;

    this->heap_insert(he);
  }

  ValueType min_abs_value() const { return this->row_heap_.empty() ? ValueType(0) : this->row_heap_.begin()->value; }

  // these are here for the unit test only
  bool validate_heap() const {
    for (int i = 0; static_cast<size_t>(i) < this->size(); i++) {
      int child0 = (i + 1) * 2 - 1;
      int child1 = (i + 1) * 2;
      if (static_cast<size_t>(child0) < this->size()
          && !less_than_abs(this->row_heap_[i].value, this->row_heap_[child0].value))
        return false;
      if (static_cast<size_t>(child1) < this->size()
          && !less_than_abs(this->row_heap_[i].value, this->row_heap_[child1].value))
        return false;
    }
    return true;
  }

  // these are here for the unit test only
  bool validate_backpointers() const {
    for (typename ainv_matrix_row::const_iterator iter = this->row_map_.begin(); iter != this->row_map_.end(); ++iter) {
      if (this->row_heap_[iter->second.heapidx].mapiter != iter
          || this->row_heap_[iter->second.heapidx].value != iter->second.value)
        return false;
    }
    return true;
  }

  void add_to_value(index_t i, ValueType addend) {
    // update val in map, which is free
    typename std::map<index_t, typename ainv_matrix_row::map_entry>::iterator map_iter = this->row_map_.find(i);
    map_iter->second.value += addend;

    // update val in heap, which requires re-sorting
    this->heap_update(map_iter->second.heapidx, map_iter->second.value);
  }

  void remove_min() {
    if (this->row_heap_.empty())
      return;

    typename std::map<index_t, typename ainv_matrix_row::map_entry>::iterator iter_to_remove
        = this->row_heap_.begin()->mapiter;
    this->heap_pop();
    this->row_map_.erase(iter_to_remove);
  }

  void replace_min_if_greater(index_t i, ValueType t) {
    if (!less_than_abs(t, this->min_abs_value())) {
      remove_min();
      insert(i, t);
    }
  }
};  // end struct ainv_matrix_row

template <typename ValueType>
void vector_scalar(std::map<index_t, ValueType> &vec, ValueType scalar) {
  for (typename std::map<index_t, ValueType>::iterator vec_iter = vec.begin(); vec_iter != vec.end(); ++vec_iter) {
    vec_iter->second *= scalar;
  }
}

template <typename ValueType>
void matrix_vector_product(basic_sparse_view<ValueType, device::cpu, sparse_format::csr> A,
                           const detail::ainv_matrix_row<ValueType> &x, std::map<index_t, ValueType> &b) {
  b.clear();

  for (typename detail::ainv_matrix_row<ValueType>::const_iterator x_iter = x.begin(); x_iter != x.end(); ++x_iter) {
    ValueType x_i = x_iter->second.value;
    index_t row = x_iter->first;

    // index_t row_start = A.row_offsets[row];
    // index_t row_end = A.row_offsets[row+1];
    index_t row_start = A.outer_ptrs()[row];
    index_t row_end = A.outer_ptrs()[row + 1];

    for (index_t row_j = row_start; row_j < row_end; row_j++) {
      // index_t col = A.column_indices[row_j];
      // ValueType Aij = A.values[row_j];
      index_t col = A.inner_indices()[row_j];
      ValueType aij = A.values()[row_j];

      ValueType product = aij * x_i;

      // add to b if it's not already in b
      typename std::map<index_t, ValueType>::iterator b_iter = b.find(col);
      if (b_iter == b.end())
        b[col] = product;
      else
        b_iter->second += product;
    }
  }
}

template <typename ValueType>
ValueType dot_product(const detail::ainv_matrix_row<ValueType> &a, const std::map<index_t, ValueType> &b) {
  typename detail::ainv_matrix_row<ValueType>::const_iterator a_iter = a.begin();
  typename std::map<index_t, ValueType>::const_iterator b_iter = b.begin();

  ValueType sum = 0;
  while (a_iter != a.end() && b_iter != b.end()) {
    index_t a_ind = a_iter->first;
    index_t b_ind = b_iter->first;
    if (a_ind == b_ind) {
      sum += a_iter->second.value * b_iter->second;
      ++a_iter;
      ++b_iter;
    } else if (a_ind < b_ind)
      ++a_iter;
    else
      ++b_iter;
  }

  return sum;
}

template <typename ValueType>
void vector_add_inplace_drop(detail::ainv_matrix_row<ValueType> &result, ValueType mult,
                             const detail::ainv_matrix_row<ValueType> &operand, ValueType tolerance,
                             int nonzeros_this_row) {
  // write into result:
  // result += mult * operand
  // but dropping any terms from (mult * operand) if they are less than tolerance

  for (typename detail::ainv_matrix_row<ValueType>::const_iterator op_iter = operand.begin(); op_iter != operand.end();
       ++op_iter) {
    index_t i = op_iter->first;
    ValueType term = mult * op_iter->second.value;
    ValueType abs_term = term < 0 ? -term : term;

    if (abs_term < tolerance)
      continue;

    // We use a combination of 2 dropping strategies: a standard drop tolerance, as well as a bound on the
    // number of non-zeros per row.  if we've already reached that maximum size
    // and this would add a new entry to result, we add it only if it is larger than one of the current entries
    // in which case we remove that element in its place.
    // This idea has been applied to IC factorization, but not to AINV as far as I'm aware.
    // See: Lin, C. and More, J. J. 1999. Incomplete Cholesky Factorizations with Limited Memory.
    //      SIAM J. Sci. Comput. 21, 1 (Aug. 1999), 24-45.

    // can improve this by storing min idx & min_abs_val for each matrix row, and keeping up to date.
    // as new entry is considered, skip if below min_val.  Otherwise, remove entry corresponding to min_val, insert new
    // entry, and search for the new min. best case, this cuts from O(n) to O(1).  Worst case stays as before. even
    // better: could i just use a heap?  i need both the map for fast inserts & deletes, and a heap to maintain lowest
    // entry this makes it O(log n) worst case, i think... idea: instead of using a map for the matrix rows, wrap it in
    // a struct that also maintains a heap of entries by abs_value
    if (result.has_entry_at_index(i))
      result.add_to_value(i, term);
    else {
      if (nonzeros_this_row < 0 || result.size() < static_cast<size_t>(nonzeros_this_row)) {
        // there is an empty slot left, so just insert
        result.insert(i, term);
      } else {
        // check if this is larger than one of the existing values.  If so, replace the smallest value.
        result.replace_min_if_greater(i, term);
      }
    }
  }
}

template <typename ValueTypeA>
basic_sparse_matrix<ValueTypeA, device::cpu, sparse_format::csr> convert_to_device_csr(
    const std::vector<detail::ainv_matrix_row<ValueTypeA>> &src) {
  // convert wt to csr
  index_t nnz = 0;
  index_t n = src.size();

  int i;
  for (i = 0; i < n; i++)
    nnz += src[i].size();

  basic_sparse_matrix<ValueTypeA, device::cpu, sparse_format::csr> host_src(n, n, nnz);
  auto row_offsets = host_src.outer_ptrs().view();
  auto col_indices = host_src.inner_indices().view();
  auto values = host_src.values().view();

  index_t pos = 0;
  // host_src.row_offsets[0] = 0;
  row_offsets[0] = 0;

  for (i = 0; i < n; i++) {
    typename detail::ainv_matrix_row<ValueTypeA>::const_iterator src_iter = src[i].begin();
    while (src_iter != src[i].end()) {
      // host_src.column_indices[pos] = src_iter->first;
      // host_src.values        [pos] = src_iter->second.value;
      col_indices[pos] = src_iter->first;
      values[pos] = src_iter->second.value;

      ++src_iter;
      ++pos;
    }
    // host_src.row_offsets[i + 1] = pos;
    row_offsets[i + 1] = pos;
  }

  // copy to device & transpose
  return host_src;
}

}  // end namespace detail

/// @brief Scaled Bridson Approximate Inverse Preconditioner
template <typename SparseBlas, typename Blas>
class bridson_ainv_preconditioner
    : public basic_preconditioner<bridson_ainv_preconditioner<SparseBlas, Blas>, typename SparseBlas::scalar_type,
                                  typename SparseBlas::device_type, SparseBlas::compression> {
public:
  using base = basic_preconditioner<bridson_ainv_preconditioner<SparseBlas, Blas>, typename SparseBlas::scalar_type,
                                    typename SparseBlas::device_type, SparseBlas::compression>;
  using Scalar = typename base::scalar_type;
  using Device = typename base::device_type;

  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  friend base;
  using const_sparse_view = sparse::basic_sparse_view<const Scalar, Device, sparse::sparse_format::csr>;
  using sparse_matrix = sparse::basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;
  // working on cpu.
  using const_cpu_view = sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr>;
  using cpu_view = sparse::basic_sparse_view<Scalar, device::cpu, sparse::sparse_format::csr>;
  using sparse_cpu_matrix = sparse::basic_sparse_matrix<Scalar, device::cpu, sparse::sparse_format::csr>;

  bridson_ainv_preconditioner() = default;
  explicit bridson_ainv_preconditioner(const const_sparse_view &view) : base(view) { this->compute({}); }

  bridson_ainv_preconditioner(bridson_ainv_preconditioner &&) = default;
  bridson_ainv_preconditioner(const bridson_ainv_preconditioner &) = delete;
  void factorize_impl() {
    auto matrix = this->matrix();
    auto n = matrix.rows(), nnz = matrix.nnz();
    sparse_cpu_matrix cpu_matrix(n, n, nnz);
    auto orig_outer = cpu_matrix.outer_ptrs().view();
    auto orig_inner = cpu_matrix.inner_indices().view();
    auto orig_values = cpu_matrix.values().view();
    // copy the matrix to cpu.
    copy(orig_outer, matrix.outer_ptrs());
    copy(orig_inner, matrix.inner_indices());
    copy(orig_values, matrix.values());

    auto h_matv = cpu_matrix.view();  // host matrix view

    // perform factorization
    std::vector<detail::ainv_matrix_row<Scalar>> w_factor(n);

    index_t i, j;
    for (i = 0; i < n; i++) {
      w_factor[i].insert(i, static_cast<Scalar>(1));
    }

    std::map<index_t, Scalar> u;
    auto h_diags_buf = make_buffer<Scalar>(n);
    auto h_diags = h_diags_buf.view();

    auto row_offsets = h_matv.outer_ptrs();

    for (j = 0; j < n; j++) {
      detail::matrix_vector_product(h_matv, w_factor[j], u);
      const Scalar p = detail::dot_product(w_factor[j], u);
      h_diags[j] = static_cast<Scalar>(1.0 / std::sqrt(p));

      // for i = j+1 to n, skipping where u_i == 0
      // this should be a O(1)-time operation, since u is a sparse vector
      for (typename std::map<index_t, Scalar>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
        i = u_iter->first;
        int row_count = nonzero_per_row_;
        if (lin_dropping_) {
          row_count = lin_param_ + static_cast<int>(row_offsets[i + 1] - row_offsets[i]);
          if (row_count < 1)
            row_count = 1;
        }
        detail::vector_add_inplace_drop(w_factor[i], -u_iter->second / p, w_factor[j], drop_tolerance_, row_count);
      }
    }

    auto w = detail::convert_to_device_csr(w_factor);
    w_ = w.to(Device());
    diags_ = h_diags_buf.to(Device());
    buffer_intern_ = make_buffer<Scalar, Device>(n);
    buffer_intern_.fill_bytes(0);
    bl_ = SparseBlas(w_.const_view());
    has_compute_ = true;
  }

private:
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(has_compute_, std::runtime_error, "The preconditioner has not been computed.");
    auto z = buffer_intern_.view();
    auto diag = diags_.const_view();
    // z = lo.T * x.
    bl_.gemv(1, x, 0, z, true);
    // z = diag * z
    Blas().emul(diag, z);
    // y = lo * y.
    bl_.gemv(1, z, 0, y, false);
  }

  Scalar drop_tolerance_ = {0.1};
  int nonzero_per_row_{-1};
  bool lin_dropping_{false};
  int lin_param_{1};
  bool has_compute_ = false;
  SparseBlas bl_;
  sparse_matrix w_;                                             // the approx inverse decomposition of A.
  contiguous_buffer<Scalar, dshape<1>, Device> diags_;          // diagonals
  contiguous_buffer<Scalar, dshape<1>, Device> buffer_intern_;  // buffer for intermediate computation.
};

/// @brief Scaled Bridson Approximate Inverse Preconditioner
template <typename SparseBlas>
class scaled_bridson_ainv_preconditioner
    : public basic_preconditioner<scaled_bridson_ainv_preconditioner<SparseBlas>, typename SparseBlas::scalar_type,
                                  typename SparseBlas::device_type, SparseBlas::compression> {
public:
  using base = basic_preconditioner<scaled_bridson_ainv_preconditioner<SparseBlas>, typename SparseBlas::scalar_type,
                                    typename SparseBlas::device_type, SparseBlas::compression>;
  using Scalar = typename base::scalar_type;
  using Device = typename base::device_type;

  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  friend base;
  using const_sparse_view = sparse::basic_sparse_view<const Scalar, Device, sparse::sparse_format::csr>;
  using sparse_matrix = sparse::basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;
  // working on cpu.
  using const_cpu_view = sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr>;
  using cpu_view = sparse::basic_sparse_view<Scalar, device::cpu, sparse::sparse_format::csr>;
  using sparse_cpu_matrix = sparse::basic_sparse_matrix<Scalar, device::cpu, sparse::sparse_format::csr>;

  scaled_bridson_ainv_preconditioner() = default;
  explicit scaled_bridson_ainv_preconditioner(const const_sparse_view &view) : base(view) { this->compute({}); }

  scaled_bridson_ainv_preconditioner(scaled_bridson_ainv_preconditioner &&) = default;
  scaled_bridson_ainv_preconditioner(const scaled_bridson_ainv_preconditioner &) = delete;
  void factorize_impl() {
    auto matrix = this->matrix();
    auto n = matrix.rows(), nnz = matrix.nnz();
    sparse_cpu_matrix cpu_matrix(n, n, nnz);
    auto orig_outer = cpu_matrix.outer_ptrs().view();
    auto orig_inner = cpu_matrix.inner_indices().view();
    auto orig_values = cpu_matrix.values().view();
    // copy the matrix to cpu.
    copy(orig_outer, matrix.outer_ptrs());
    copy(orig_inner, matrix.inner_indices());
    copy(orig_values, matrix.values());

    auto h_matv = cpu_matrix.view();  // host matrix view

    // perform factorization
    std::vector<detail::ainv_matrix_row<Scalar>> w_factor(n);

    index_t i, j;
    for (i = 0; i < n; i++) {
      w_factor[i].insert(i, static_cast<Scalar>(1));
    }

    std::map<index_t, Scalar> u;

    auto row_offsets = h_matv.outer_ptrs();

    for (j = 0; j < n; j++) {
      detail::matrix_vector_product(h_matv, w_factor[j], u);
      const Scalar p = detail::dot_product(w_factor[j], u);
      const Scalar rsqrt_p = static_cast<Scalar>(1.0 / std::sqrt(p));
      detail::vector_scalar(u, rsqrt_p);
      w_factor[j].mult_by_scalar(rsqrt_p);

      // for i = j+1 to n, skipping where u_i == 0
      // this should be a O(1)-time operation, since u is a sparse vector
      for (typename std::map<index_t, Scalar>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
        i = u_iter->first;
        int row_count = nonzero_per_row_;
        if (lin_dropping_) {
          row_count = lin_param_ + static_cast<int>(row_offsets[i + 1] - row_offsets[i]);
          if (row_count < 1)
            row_count = 1;
        }
        detail::vector_add_inplace_drop(w_factor[i], -u_iter->second, w_factor[j], drop_tolerance_, row_count);
      }
    }

    auto w = detail::convert_to_device_csr(w_factor);

    w_ = w.to(Device());
    buffer_intern_ = make_buffer<Scalar, Device>(n);
    buffer_intern_.fill_bytes(0);
    bl_ = SparseBlas(w_.const_view());
    has_compute_ = true;
  }

private:
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(has_compute_, std::runtime_error, "The preconditioner has not been computed.");
    auto z = buffer_intern_.view();
    // z = lo.T * x.
    bl_.gemv(1, x, 0, z, true);
    // y = lo * y.
    bl_.gemv(1, z, 0, y, false);
  }

  Scalar drop_tolerance_ = {0.1};
  int nonzero_per_row_{-1};
  bool lin_dropping_{false};
  int lin_param_{1};
  bool has_compute_ = false;
  SparseBlas bl_;
  sparse_matrix w_;                                             // the approx inverse decomposition of A.
  contiguous_buffer<Scalar, dshape<1>, Device> buffer_intern_;  // buffer for intermediate computation.
};

}  // namespace mathprim::sparse::iterative
