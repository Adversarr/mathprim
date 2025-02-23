#pragma once
#include <sstream>
#include <vector>

#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/supports/view_from/stl.hpp"
namespace mathprim::io {

template <typename Scalar>
class matrix_market {
public:
  using coo_matrix = sparse::basic_sparse_matrix<Scalar, device::cpu, sparse::sparse_format::coo>;
  using const_view = sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::coo>;

  void write(std::ostream& os, const const_view& view);

  coo_matrix read(std::istream& is);
};

template <typename Scalar>
void matrix_market<Scalar>::write(std::ostream& os, const const_view& view) {
  // Write header
  os << "%%MatrixMarket matrix coordinate real general\n";

  // Write comments (optional)
  os << "%\n";  // No comments for now

  // Write dimensions and number of non-zero entries
  auto shape = view.shape();
  os << shape[0] << " " << shape[1] << " " << view.nnz() << "\n";

  // Write matrix data (COO format)
  auto row_indices = view.outer_ptrs();
  auto col_indices = view.inner_indices();
  auto values = view.values();

  for (index_t i = 0; i < view.nnz(); ++i) {
    os << (row_indices[i] + 1) << " " << (col_indices[i] + 1) << " " << values[i] << "\n";
  }
}

template <typename Scalar>
typename matrix_market<Scalar>::coo_matrix matrix_market<Scalar>::read(std::istream& is) {
  std::string line;

  // Read header
  std::getline(is, line);
  if (line.find("%%MatrixMarket matrix coordinate real general") == std::string::npos) {
    throw std::runtime_error("Invalid Matrix Market file: incorrect header");
  }

  // Skip comments
  while (std::getline(is, line) && line[0] == '%') {
  }

  // Read dimensions and number of non-zeros
  index_t rows, cols, nnz;
  std::istringstream(line) >> rows >> cols >> nnz;

  // Read matrix data
  std::vector<index_t> row_indices(nnz), col_indices(nnz);
  std::vector<Scalar> values(nnz);

  for (index_t i = 0; i < nnz; ++i) {
    std::getline(is, line);
    std::istringstream iss(line);
    iss >> row_indices[i] >> col_indices[i] >> values[i];
    // Convert from 1-based to 0-based indexing
    row_indices[i]--;
    col_indices[i]--;
  }

  // Construct and return COO matrix
  coo_matrix mat(rows, cols, nnz);
  copy(mat.outer_ptrs().view(), view(row_indices));
  copy(mat.inner_indices().view(), view(col_indices));
  copy(mat.values().view(), view(values));

  return mat;
}
}  // namespace mathprim::io