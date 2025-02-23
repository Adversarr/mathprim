#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"

namespace mathprim::iterative_solver {

namespace internal {

template <typename Scalar, typename Device, sparse::sparse_format Compression>
struct diagonal_extract;

template <typename Scalar>
struct diagonal_extract<Scalar, device::cpu, sparse::sparse_format::csr> {
  using buffer_type = continuous_buffer<Scalar, shape_t<keep_dim>, device::cpu>;

  static buffer_type extract(
      const sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr>& mat) {
    auto diag = make_buffer<Scalar, device::cpu>(make_shape(mat.rows()));
    auto row_ptr = mat.outer_ptrs();
    auto col_idx = mat.inner_indices();
    auto values = mat.values();
    auto dv = diag.view();
    for (index_t i = 0; i < mat.rows(); ++i) {
      bool found = false;
      for (index_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        if (col_idx[j] == i) {
          dv[i] = static_cast<Scalar>(1) / values[j];
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error("The diagonal element is not found for row " + std::to_string(i) + ".");
      }
    }
    return diag;
  }
};

template <typename Scalar>
struct diagonal_extract<Scalar, device::cpu, sparse::sparse_format::csc> {
  using buffer_type = continuous_buffer<Scalar, shape_t<keep_dim>, device::cpu>;

  static buffer_type extract(
      const sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr>& mat) {
    auto diag = make_buffer<Scalar, device::cpu>(make_shape(mat.cols()));
    auto col_ptr = mat.outer_ptrs();
    auto row_idx = mat.inner_indices();
    auto values = mat.values();
    auto dv = diag.view();

    for (index_t i = 0; i < mat.cols(); ++i) {
      bool found = false;
      for (index_t j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
        if (row_idx[j] == i) {
          dv[i] = static_cast<Scalar>(1) / values[j];
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error("The diagonal element is not found for row " + std::to_string(i) + ".");
      }
    }
    return diag;
  }
};

}  // namespace internal

template <typename Scalar, typename Device, sparse::sparse_format Compression, typename Blas>
class diagonal_preconditioner
    : public basic_preconditioner<diagonal_preconditioner<Scalar, Device, Compression, Blas>, Scalar, Device> {
public:
  using const_sparse = sparse::basic_sparse_view<const Scalar, Device, Compression>;
  using buffer_type = continuous_buffer<Scalar, shape_t<keep_dim>, Device>;
  using base = basic_preconditioner<diagonal_preconditioner<Scalar, Device, Compression, Blas>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;

  explicit diagonal_preconditioner(const_sparse sparse_matrix) :
      inv_diag_(internal::diagonal_extract<Scalar, Device, Compression>::extract(sparse_matrix)) {}
  diagonal_preconditioner(diagonal_preconditioner&&) = default;

  // Y <- D^-1 * X
  void apply_impl(vector_type y, const_vector x) {
    blas_.copy(y, x); // Y = X
    blas_.emul(inv_diag_.const_view(), y);
  }

private:
  buffer_type inv_diag_;
  Blas blas_;
};

}  // namespace mathprim::iterative_solver

///////////////////////////////////////////////////////////////////////////////
/// CUDA implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__
#  include "mathprim/parallel/cuda.cuh"

namespace mathprim::iterative_solver::internal {
template <typename Scalar>
struct diagonal_extract<Scalar, device::cuda, sparse::sparse_format::csr> {
  using buffer_type = continuous_buffer<Scalar, shape_t<keep_dim>, device::cuda>;

  static buffer_type extract(
      const sparse::basic_sparse_view<const Scalar, device::cuda, sparse::sparse_format::csr>& mat) {
    auto diag = make_cuda_buffer<Scalar>(make_shape(mat.rows()));
    par::cuda pf;

    pf.run(make_shape(mat.rows()), [row_ptr = mat.outer_ptrs(), col_idx = mat.inner_indices(), values = mat.values(),
                                    diag = diag.view()] __device__(index_t i) {
      for (index_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        if (col_idx[j] == i) {
          diag[i] = static_cast<Scalar>(1) / values[j];
          return;
        }
      }
      printf("DiagonalPreconditioner(CUDA): Failed to find the diagonal element for row %d!!!\n", i);
    });

    return diag;
  }
};

template <typename Scalar>
struct diagonal_extract<Scalar, device::cuda, sparse::sparse_format::csc> {
  using buffer_type = continuous_buffer<Scalar, shape_t<keep_dim>, device::cuda>;

  static buffer_type extract(
      const sparse::basic_sparse_view<const Scalar, device::cuda, sparse::sparse_format::csc>& mat) {
    auto diag = make_cuda_buffer<Scalar>(make_shape(mat.rows()));
    par::cuda pf;

    pf.run(make_shape(mat.rows()), [col_ptr = mat.outer_ptrs(), row_idx = mat.inner_indices(), values = mat.values(),
                                    diag = diag.view()] __device__(index_t i) {
      for (index_t j = col_ptr[i]; j < col_ptr[i + 1]; ++j) {
        if (row_idx[j] == i) {
          diag[i] = static_cast<Scalar>(1) / values[j];
          return;
        }
      }
      printf("DiagonalPreconditioner(CUDA): Failed to find the diagonal element for row %d!!!\n", i);
    });

    return diag;
  }
};
}  // namespace mathprim::iterative_solver::internal
#endif