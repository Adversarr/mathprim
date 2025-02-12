#pragma once

#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

namespace mathprim::sparse {
namespace blas {
template <typename Scalar, sparse_format sparse_compression>
class eigen : public sparse_blas_base<Scalar, device::cpu, sparse_compression> {
public:
  using base = sparse_blas_base<Scalar, device::cpu, sparse_compression>;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using base::base;

  // y = alpha * A * x + beta * y.
  void gemv(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) override {
    this->check_gemv_shape(x, y);
    bool transpose = false;

    if (this->mat_.is_transpose()) {  // Computes A.T @ x
      if (this->mat_.property() == sparse_property::symmetric) {
        // Symmetric matrix, use the same code path for both transposed and non-transposed.
        transpose = false;
      } else if (this->mat_.property() == sparse_property::skew_symmetric) {
        // A = -A.T => A.T @ x = -A @ x
        transpose = false;
        alpha = -alpha;
      }
      transpose = true;
    }

    auto mat = eigen_support::map(this->mat_);
    auto x_map = eigen_support::cmap(x);
    auto y_map = eigen_support::cmap(y);
    if (transpose) {
      y_map = alpha * mat.transpose() * x_map + beta * y_map;
    } else {
      y_map = alpha * mat * x_map + beta * y_map;
    }
  }
};

}  // namespace blas
}  // namespace mathprim::sparse