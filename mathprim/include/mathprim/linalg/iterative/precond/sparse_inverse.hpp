#pragma once
#include <cmath>

#include "mathprim/core/buffer.hpp"
#include "mathprim/linalg/iterative/internal/diagonal_extract.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

namespace mathprim::sparse::iterative {

template <typename SparseBlas, typename Blas>
class sparse_preconditioner
    : public basic_preconditioner<sparse_preconditioner<SparseBlas, Blas>, typename SparseBlas::scalar_type,
                                  typename SparseBlas::device_type, SparseBlas::compression> {
public:
  using base = basic_preconditioner<sparse_preconditioner<SparseBlas, Blas>, typename SparseBlas::scalar_type,
                                    typename SparseBlas::device_type, SparseBlas::compression>;
  using Scalar = typename base::scalar_type;
  using Device = typename base::device_type;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = typename base::const_sparse;
  using sparse_view = sparse::basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;

  friend base;

  sparse_preconditioner() = default;
  explicit sparse_preconditioner(const_sparse mat) : base(mat) {}

  sparse_preconditioner(const sparse_preconditioner&) = delete;
  sparse_preconditioner(sparse_preconditioner&&) = default;

  using sparse_matrix = basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;

  void set_approximation(const_sparse mat, Scalar eps) {
    bl_l_ = SparseBlas(mat);
    buffer_intern_ = make_buffer<Scalar, Device>(mat.rows());
    eps_ = eps;
  }

  void set_approximation(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor, index_t>& mat, Scalar eps) {
    eps_ = eps;
    MATHPRIM_INTERNAL_CHECK_THROW(mat.isCompressed(), std::runtime_error,  //
                                  "Eigen sparse matrix must be compressed.");

    Eigen::SparseMatrix<Scalar, Eigen::RowMajor, index_t> mat_transpose = mat.transpose();
    mat_transpose.makeCompressed();
    auto matview = eigen_support::view(mat);
    auto matview_transpose = eigen_support::view(mat_transpose);

    // copy the matrix locally.
    mat_l_ = sparse_matrix(matview.rows(), matview.cols(), matview.nnz());
    mat_u_ = sparse_matrix(matview_transpose.rows(), matview_transpose.cols(), matview_transpose.nnz());

    copy(mat_l_.view().outer_ptrs(), matview.outer_ptrs());
    copy(mat_l_.view().inner_indices(), matview.inner_indices());
    copy(mat_l_.view().values(), matview.values());
    copy(mat_u_.view().outer_ptrs(), matview_transpose.outer_ptrs());
    copy(mat_u_.view().inner_indices(), matview_transpose.inner_indices());
    copy(mat_u_.view().values(), matview_transpose.values());
    buffer_intern_ = make_buffer<Scalar, Device>(mat.rows());
    bl_l_ = SparseBlas(mat_l_.view());
    bl_u_ = SparseBlas(mat_u_.view());
  }

private:
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(buffer_intern_, std::runtime_error, "Preconditioner not initialized.");
    // z = lo.T * x.
    auto z = buffer_intern_.view();
    if (mat_u_) {
      bl_u_.gemv(1, x, 0, z, false);
    } else {
      bl_l_.gemv(1, x, 0, z, true);
    }

    // y = lo * y.
    bl_l_.gemv(1, z, 0, y, false);
    // y = y + eps x
    dense_bl_.axpy(eps_, x, y);
  }

  Scalar eps_;
  Blas dense_bl_;
  contiguous_vector_buffer<Scalar, Device> buffer_intern_;

  SparseBlas bl_l_;
  SparseBlas bl_u_;
  sparse_matrix mat_l_;
  sparse_matrix mat_u_;
};

template <typename SparseBlas, typename Blas>
class scale_sparse_preconditioner
    : public basic_preconditioner<scale_sparse_preconditioner<SparseBlas, Blas>, typename SparseBlas::scalar_type,
                                  typename SparseBlas::device_type, SparseBlas::compression> {
public:
  using base = basic_preconditioner<scale_sparse_preconditioner<SparseBlas, Blas>, typename SparseBlas::scalar_type,
                                    typename SparseBlas::device_type, SparseBlas::compression>;
  using Scalar = typename base::scalar_type;
  using Device = typename base::device_type;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = typename base::const_sparse;
  using sparse_view = sparse::basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;

  friend base;

  scale_sparse_preconditioner() = default;
  explicit scale_sparse_preconditioner(const_sparse mat) : base(mat) {}

  scale_sparse_preconditioner(const scale_sparse_preconditioner&) = delete;
  scale_sparse_preconditioner(scale_sparse_preconditioner&&) = default;

  using sparse_matrix = basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;

  void set_approximation(const_sparse mat, Scalar eps, bool already_scaled=false) {
    already_scaled_ = already_scaled;
    bl_l_ = SparseBlas(mat);
    buffer_intern_ = make_buffer<Scalar, Device>(mat.rows());
    eps_ = eps;
  }

  void set_approximation(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor, index_t>& mat, Scalar eps, bool already_scaled=false) {
    already_scaled_ = already_scaled;
    eps_ = eps;
    MATHPRIM_INTERNAL_CHECK_THROW(mat.isCompressed(), std::runtime_error,  //
                                  "Eigen sparse matrix must be compressed.");

    Eigen::SparseMatrix<Scalar, Eigen::RowMajor, index_t> mat_transpose = mat.transpose();
    mat_transpose.makeCompressed();
    auto matview = eigen_support::view(mat);
    auto matview_transpose = eigen_support::view(mat_transpose);

    // copy the matrix locally.
    mat_l_ = sparse_matrix(matview.rows(), matview.cols(), matview.nnz());
    mat_u_ = sparse_matrix(matview_transpose.rows(), matview_transpose.cols(), matview_transpose.nnz());

    copy(mat_l_.view().outer_ptrs(), matview.outer_ptrs());
    copy(mat_l_.view().inner_indices(), matview.inner_indices());
    copy(mat_l_.view().values(), matview.values());
    copy(mat_u_.view().outer_ptrs(), matview_transpose.outer_ptrs());
    copy(mat_u_.view().inner_indices(), matview_transpose.inner_indices());
    copy(mat_u_.view().values(), matview_transpose.values());
    buffer_intern_ = make_buffer<Scalar, Device>(mat.rows());
    bl_l_ = SparseBlas(mat_l_.view());
    bl_u_ = SparseBlas(mat_u_.view());
  }

  void factorize_impl() {
    inv_diag_ = internal::diagonal_extract<Scalar, Device, sparse_format::csr>::extract(this->matrix());
  }

private:
  // computes:
  // y = (eps D^-1 + L D^-1 U) y
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(buffer_intern_, std::runtime_error, "Preconditioner not initialized.");
    auto invd = inv_diag_.const_view();
    // copy(y, x);     // y = x
    // dense_bl_.inplace_emul(invd, y);  // y = D^-1 * x
    dense_bl_.emul(invd, x, y);  // y = D^-1 * x

    // z = lo.T * x.
    auto z = buffer_intern_.view();
    if (mat_u_) {
      bl_u_.gemv(1, x, 0, z, false);
    } else {
      bl_l_.gemv(1, x, 0, z, true);
    }

    if (!already_scaled_) {
      dense_bl_.inplace_emul(invd, z);
    }

    // y = lo * z + eps_ * y
    bl_l_.gemv(1, z, eps_, y, false);
  }

  Scalar eps_;
  Blas dense_bl_;
  contiguous_vector_buffer<Scalar, Device> buffer_intern_;
  contiguous_vector_buffer<Scalar, Device> inv_diag_;

  SparseBlas bl_l_;
  SparseBlas bl_u_;
  sparse_matrix mat_l_;
  sparse_matrix mat_u_;
  bool already_scaled_ = false;
};

}  // namespace mathprim::sparse::iterative
