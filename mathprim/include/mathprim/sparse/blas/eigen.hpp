#pragma once

#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

namespace mathprim::sparse {
namespace blas {
template <typename Scalar, sparse_format SparseCompression>
class eigen : public sparse_blas_base<eigen<Scalar, SparseCompression>, Scalar, device::cpu, SparseCompression> {
public:
  using base = sparse_blas_base<eigen<Scalar, SparseCompression>, Scalar, device::cpu, SparseCompression>;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using base::base;
  friend base;

protected:
  // C <- alpha op(A) B + beta C => C.T <- beta C.T + alpha B.T op(A).T
  template <typename SshapeB, typename SstrideB, typename SshapeC, typename SstrideC>
  void spmm_impl(Scalar alpha, basic_view<const Scalar, SshapeB, SstrideB, device::cpu> B, Scalar beta,
                 basic_view<Scalar, SshapeC, SstrideC, device::cpu> C, bool transA = false) {
    auto mat = eigen_support::map(this->mat_);
    if (B.is_contiguous() && C.is_contiguous()) {
      auto b_map = eigen_support::cmap(B).transpose();
      auto c_map = eigen_support::cmap(C).transpose();
      if (transA) {
        c_map = alpha * mat.transpose() * b_map + beta * c_map;
      } else {
        c_map = alpha * mat * b_map + beta * c_map;
      }
    } else {
      auto b_map = eigen_support::map(B).transpose();
      auto c_map = eigen_support::map(C).transpose();
      if (transA) {
        c_map = alpha * mat.transpose() * b_map + beta * c_map;
      } else {
        c_map = alpha * mat * b_map + beta * c_map;
      }
    }
  }

  // y = alpha * A * x + beta * y.
  void gemv_impl(Scalar alpha, const_vector_view x, Scalar beta, vector_view y, bool transpose) {
    bool transpose_actual = false;

    if (transpose) {  // Computes A.T @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        transpose_actual = false;
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        transpose_actual = false;
        alpha = -alpha;
      } else {
        transpose_actual = true;
      }
    }

    auto mat = eigen_support::map(this->mat_);
    auto x_map = eigen_support::cmap(x);
    auto y_map = eigen_support::cmap(y);
    if (beta == 0) {
      y_map.setZero();
    } else {
      y_map = y_map * beta;
    }
    if (transpose_actual) {
      y_map += alpha * mat.transpose() * x_map;
    } else {
      y_map += alpha * mat * x_map;
    }
  }
};

}  // namespace blas
}  // namespace mathprim::sparse