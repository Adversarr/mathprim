#pragma once
#include <Eigen/Sparse>

#include "mathprim/core/utils/common.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::eigen_support {

template <typename Scalar, int Options>
using EigenSparseMatrix = Eigen::SparseMatrix<Scalar, Options, index_t>;

namespace internal {

template <typename Scalar, sparse::sparse_format sparse_compression>
struct eigen_sparse_format;

template <typename Scalar>
struct eigen_sparse_format<Scalar, sparse::sparse_format::csr> {
  using type = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
};

template <typename Scalar>
struct eigen_sparse_format<Scalar, sparse::sparse_format::csc> {
  using type = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
};

template <typename Scalar, sparse::sparse_format sparse_compression>
using eigen_sparse_format_t = typename eigen_sparse_format<Scalar, sparse_compression>::type;

template <typename SparseMatrixT>
struct mp_sparse_format {
  static_assert(::mathprim::internal::always_false_v<SparseMatrixT>, "Unsupported type.");
};

template <typename Scalar>
struct mp_sparse_format<EigenSparseMatrix<Scalar, Eigen::RowMajor>> {
  static constexpr sparse::sparse_format value = sparse::sparse_format::csr;
};

template <typename Scalar>
struct mp_sparse_format<EigenSparseMatrix<Scalar, Eigen::ColMajor>> {
  static constexpr sparse::sparse_format value = sparse::sparse_format::csc;
};

template <typename SparseMatrixT>
constexpr sparse::sparse_format mp_sparse_format_v = mp_sparse_format<SparseMatrixT>::value;

}  // namespace internal

template <typename Scalar, sparse::sparse_format sparse_compression>
Eigen::Map<internal::eigen_sparse_format_t<Scalar, sparse_compression>> map(
    sparse::basic_sparse_view<Scalar, device::cpu, sparse_compression, false> mat) {
  using EigenSparseMatrix = internal::eigen_sparse_format_t<Scalar, sparse_compression>;
  auto rows = mat.rows();
  auto cols = mat.cols();
  auto nnz = mat.nnz();
  auto* outer_ptrs = mat.outer_ptrs().data();
  auto* inner_indices = mat.inner_indices().data();
  auto* values = mat.values().data();
  return Eigen::Map<EigenSparseMatrix>(rows, cols, nnz, outer_ptrs, inner_indices, values);
}

template <typename Scalar, sparse::sparse_format sparse_compression>
Eigen::Map<const internal::eigen_sparse_format_t<Scalar, sparse_compression>> map(
    sparse::basic_sparse_view<Scalar, device::cpu, sparse_compression, true> mat) {
  using EigenSparseMatrix = internal::eigen_sparse_format_t<Scalar, sparse_compression>;
  auto rows = mat.rows();
  auto cols = mat.cols();
  auto nnz = mat.nnz();
  auto* outer_ptrs = mat.outer_ptrs().data();
  auto* inner_indices = mat.inner_indices().data();
  auto* values = mat.values().data();
  return Eigen::Map<const EigenSparseMatrix>(rows, cols, nnz, outer_ptrs, inner_indices, values);
}

template <typename dev = device::cpu, typename Scalar, int Options>
auto view(EigenSparseMatrix<Scalar, Options>& mat, sparse::sparse_property property = sparse::sparse_property::general) {
  using SparseMatrixT = EigenSparseMatrix<Scalar, Options>;
  constexpr auto sparse_format = internal::mp_sparse_format_v<SparseMatrixT>;
  using RetT = sparse::basic_sparse_view<Scalar, dev, sparse_format, false>;
  if (!mat.isCompressed()) {
    throw std::runtime_error("Eigen sparse matrix must be compressed.");
  }


  auto rows = mat.rows();
  auto cols = mat.cols();
  auto nnz = mat.nonZeros();
  auto n_outer = mat.outerSize();
  auto outer_ptrs = ::mathprim::view<dev>(mat.outerIndexPtr(), make_shape(n_outer + 1));
  auto inner_indices = ::mathprim::view<dev>(mat.innerIndexPtr(), make_shape(nnz));
  auto values = ::mathprim::view<dev>(mat.valuePtr(), make_shape(nnz));
  return RetT(values, outer_ptrs, inner_indices, rows, cols, nnz, property, false);
}

template <typename dev = device::cpu, typename Scalar, int Options>
auto view(const EigenSparseMatrix<Scalar, Options>& mat, sparse::sparse_property property = sparse::sparse_property::general) {
  using SparseMatrixT = EigenSparseMatrix<Scalar, Options>;
  constexpr auto sparse_format = internal::mp_sparse_format_v<SparseMatrixT>;
  using RetT = sparse::basic_sparse_view<Scalar, dev, sparse_format, true>;
  if (!mat.isCompressed()) {
    throw std::runtime_error("Eigen sparse matrix must be compressed.");
  }

  auto rows = mat.rows();
  auto cols = mat.cols();
  auto nnz = mat.nonZeros();
  auto n_outer = mat.outerSize();
  auto outer_ptrs = ::mathprim::view<dev>(mat.outerIndexPtr(), make_shape(n_outer + 1));
  auto inner_indices = ::mathprim::view<dev>(mat.innerIndexPtr(), make_shape(nnz));
  auto values = ::mathprim::view<dev>(mat.valuePtr(), make_shape(nnz));
  return RetT(values, outer_ptrs, inner_indices, rows, cols, nnz, property, true);
}

}  // namespace mathprim::eigen_support