#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

#ifdef __CUDACC__
#  include "mathprim/parallel/cuda.cuh"
#endif
namespace mathprim::sparse::iterative::internal {

template <typename Scalar, typename Device, sparse::sparse_format Compression>
struct diagonal_extract;

template <typename Scalar, sparse::sparse_format Compression>
struct diagonal_extract<Scalar, device::cpu, Compression> {
  using buffer_type = contiguous_buffer<Scalar, shape_t<keep_dim>, device::cpu>;

  static buffer_type extract(const sparse::basic_sparse_view<const Scalar, device::cpu, Compression>& mat) {
    auto diag = make_buffer<Scalar, device::cpu>(make_shape(mat.rows()));
    auto row_ptr = mat.outer_ptrs();
    auto col_idx = mat.inner_indices();
    auto values = mat.values();
    auto dv = diag.view();
    diag.fill_bytes(0);
    visit(mat, par::seq(), [&](index_t i, index_t j, Scalar val) {
      if (i == j) {
        dv[i] = static_cast<Scalar>(1) / val;
      }
    });
    for (index_t i = 0; i < mat.rows(); ++i) {
      MATHPRIM_INTERNAL_CHECK_THROW(dv[i] != 0, std::runtime_error,
                                    "The diagonal element is not found for row " + std::to_string(i) + ".");
    }
    return diag;
  }
};

///////////////////////////////////////////////////////////////////////////////
/// CUDA implementation
///////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
template <typename Scalar>
struct diagonal_extract<Scalar, device::cuda, sparse::sparse_format::csr> {
  using buffer_type = contiguous_buffer<Scalar, shape_t<keep_dim>, device::cuda>;

  static buffer_type extract(
      const sparse::basic_sparse_view<const Scalar, device::cuda, sparse::sparse_format::csr>& mat) {
    auto diag = make_cuda_buffer<Scalar>(make_shape(mat.rows()));
    par::cuda pf;
    diag.fill_bytes(0);
    visit(mat, pf, [dv = diag.view()] __device__(index_t i, index_t j, Scalar val) {
      if (i == j) {
        dv[i] = static_cast<Scalar>(1) / val;
      }
    });
    pf.run(make_shape(mat.rows()), [diag = diag.view()] __device__(index_t i) {
      if (diag[i] == 0) {
        printf("DiagonalPreconditioner(CUDA): Failed to find the diagonal element for row %d!!!\n", i);
      }
    });
    return diag;
  }
};
#endif
}  // namespace mathprim::sparse::iterative::internal
