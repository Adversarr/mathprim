#pragma once
#include <algorithm>
#include <vector>

#include "basic_sparse.hpp"
namespace mathprim::sparse {

template <typename Scalar, sparse_format SparseCompression, typename Device>
struct format_convert {
  static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported conversion");

  using matrix_type = basic_sparse_matrix<Scalar, Device, SparseCompression>;
  using const_view_type = basic_sparse_view<const Scalar, Device, SparseCompression>;
  using coo_matrix = basic_sparse_matrix<Scalar, Device, sparse_format::coo>;
  using const_coo_view = basic_sparse_view<const Scalar, Device, sparse_format::coo>;
  using dense_type = continuous_buffer<Scalar, dshape<2>, Device>;


  // API for converting from one format to another
  static coo_matrix to_coo(const const_view_type& matrix);
  static matrix_type from_coo(const const_coo_view& coo);
  static dense_type to_dense(const const_view_type& matrix);
};

template <typename Scalar>
struct format_convert<Scalar, sparse_format::csr, device::cpu> {
  using matrix_type = basic_sparse_matrix<Scalar, device::cpu, sparse_format::csr>;
  using const_view_type = basic_sparse_view<const Scalar, device::cpu, sparse_format::csr>;
  using coo_matrix = basic_sparse_matrix<Scalar, device::cpu, sparse_format::coo>;
  using const_coo_view = basic_sparse_view<const Scalar, device::cpu, sparse_format::coo>;
  static MATHPRIM_NOINLINE coo_matrix to_coo(const const_view_type& csr) {
    auto [rows, cols] = csr.shape();
    auto nnz = csr.nnz();
    coo_matrix result(rows, cols, nnz, csr.property());

    auto outer_ptrs = csr.outer_ptrs();        // rowptrs
    auto inner_indices = csr.inner_indices();  // colidx
    auto values = csr.values();

    auto result_outer_ptrs = result.outer_ptrs().view();        // row
    auto result_inner_indices = result.inner_indices().view();  // col
    auto result_values = result.values().view();

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = outer_ptrs[i]; j < outer_ptrs[i + 1]; ++j) {
        result_outer_ptrs[j] = i;
        result_inner_indices[j] = inner_indices[j];
        result_values[j] = values[j];
      }
    }

    return result;
  }

  static MATHPRIM_NOINLINE matrix_type from_coo(const const_coo_view& coo) {
    auto [rows, cols] = coo.shape();
    auto nnz = coo.nnz();
    matrix_type result(rows, cols, nnz, coo.property());

    auto outer_ptrs = coo.outer_ptrs();        // rowptrs
    auto inner_indices = coo.inner_indices();  // colidx
    auto values = coo.values();

    auto result_outer_ptrs = result.outer_ptrs().view();
    auto result_inner_indices = result.inner_indices().view();
    auto result_values = result.values().view();
    result.outer_ptrs().fill_bytes(0);

    // Count the number of non-zero elements per row
    for (index_t i = 0; i < nnz; ++i) {
      result_outer_ptrs[outer_ptrs[i]]++;
    }

    // Cumulative sum to get the correct row pointers
    for (index_t i = 1; i < rows; ++i) {
      result_outer_ptrs[i] += result_outer_ptrs[i - 1];
    }

    // Shift row pointers to the right
    for (index_t i = rows; i > 0; --i) {
      result_outer_ptrs[i] = result_outer_ptrs[i - 1];
    }
    result_outer_ptrs[0] = 0;

    // Fill the CSR matrix.
    std::vector<index_t> counter(rows, 0);
    for (index_t i = 0; i < nnz; ++i) {
      index_t row = outer_ptrs[i];
      index_t dest = result_outer_ptrs[row] + counter[row]++;
      result_inner_indices[dest] = inner_indices[i];
      result_values[dest] = values[i];
    }

    return result;
  }
};

template <typename Scalar>
struct format_convert<Scalar, sparse_format::csc, device::cpu> {
  using matrix_type = basic_sparse_matrix<Scalar, device::cpu, sparse_format::csc>;
  using const_view_type = basic_sparse_view<const Scalar, device::cpu, sparse_format::csc>;
  using coo_matrix = basic_sparse_matrix<Scalar, device::cpu, sparse_format::coo>;
  using coo_const_view = basic_sparse_view<const Scalar, device::cpu, sparse_format::coo>;

  static MATHPRIM_NOINLINE coo_matrix to_coo(const const_view_type& csc) {
    auto [rows, cols] = csc.shape();
    auto nnz = csc.nnz();
    coo_matrix result(rows, cols, nnz, csc.property());

    auto outer_ptrs = csc.outer_ptrs();        // colptrs
    auto inner_indices = csc.inner_indices();  // rowidx
    auto values = csc.values();

    auto result_outer_ptrs = result.outer_ptrs().view();        // row
    auto result_inner_indices = result.inner_indices().view();  // col
    auto result_values = result.values().view();

    for (index_t j = 0; j < cols; ++j) {
      for (index_t i = outer_ptrs[j]; i < outer_ptrs[j + 1]; ++i) {
        result_outer_ptrs[i] = inner_indices[i];
        result_inner_indices[i] = j;
        result_values[i] = values[i];
      }
    }

    return result;
  }

  static MATHPRIM_NOINLINE matrix_type from_coo(const coo_const_view& coo) {
    auto [rows, cols] = coo.shape();
    auto nnz = coo.nnz();
    matrix_type result(rows, cols, nnz, coo.property());

    auto outer_ptrs = coo.outer_ptrs();        // row indices
    auto inner_indices = coo.inner_indices();  // col indices
    auto values = coo.values();

    auto result_outer_ptrs = result.outer_ptrs().view();
    auto result_inner_indices = result.inner_indices().view();
    auto result_values = result.values().view();
    result.outer_ptrs().fill_bytes(0);

    // Count the number of non-zero elements per column
    for (index_t i = 0; i < nnz; ++i) {
      result_outer_ptrs[inner_indices[i]]++;
    }

    // Cumulative sum to get the correct column pointers
    for (index_t i = 1; i < cols; ++i) {
      result_outer_ptrs[i] += result_outer_ptrs[i - 1];
    }

    // Shift column pointers to the right
    for (index_t i = cols; i > 0; --i) {
      result_outer_ptrs[i] = result_outer_ptrs[i - 1];
    }
    result_outer_ptrs[0] = 0;

    // Fill the CSC matrix.
    std::vector<index_t> counter(cols, 0);
    for (index_t i = 0; i < nnz; ++i) {
      index_t col = inner_indices[i];
      index_t dest = result_outer_ptrs[col] + counter[col]++;
      result_inner_indices[dest] = outer_ptrs[i];
      result_values[dest] = values[i];
    }

    return result;
  }
};

template <typename Scalar, sparse_format TargetFormat>
auto make_from_coos(const basic_sparse_matrix<Scalar, device::cpu, sparse_format::coo>& coo) {
  return format_convert<Scalar, TargetFormat, device::cpu>::from_coo(coo.view());
}

template <typename Scalar, typename Iter>
basic_sparse_matrix<Scalar, device::cpu, sparse_format::coo> make_from_triplets(Iter begin, Iter end, index_t rows,
                                                                                index_t cols) {
  using coo_matrix = basic_sparse_matrix<Scalar, device::cpu, sparse_format::coo>;
  std::vector<sparse_entry<Scalar>> merged(begin, end);

  // sort the triplets by row and column
  std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) -> bool {
    return a.row_ < b.row_ || (a.row_ == b.row_ && a.col_ < b.col_);
  });

  // merge the triplets with the same row and column
  auto it_out = merged.begin();
  auto it_in = it_out;
  for (++it_in; it_in != merged.end(); ++it_in) {
    if (it_out->row_ == it_in->row_ && it_out->col_ == it_in->col_) {
      it_out->value_ += it_in->value_;
    } else {
      ++it_out;
      *it_out = *it_in;
    }
  }

  // create the COO matrix
  auto nnz = static_cast<index_t>(std::distance(merged.begin(), it_out) + 1);
  coo_matrix result(rows, cols, nnz, sparse_property::general);
  auto outer_ptrs = result.outer_ptrs().view();
  auto inner_indices = result.inner_indices().view();
  auto values = result.values().view();
  for (index_t i = 0; i < nnz; ++i) {
    outer_ptrs[i] = merged[i].row_;
    inner_indices[i] = merged[i].col_;
    values[i] = merged[i].value_;
  }

  return result;
}

template <typename Scalar>
void print(std::ostream& os, const basic_sparse_view<Scalar, device::cpu, sparse_format::coo>& coo) {
  auto outer_ptrs = coo.outer_ptrs();
  auto inner_indices = coo.inner_indices();
  auto values = coo.values();
  for (index_t i = 0; i < coo.nnz(); ++i) {
    os << "(" << outer_ptrs[i] << ", " << inner_indices[i] << ") = " << values[i] << std::endl;
  }
}

}  // namespace mathprim::sparse