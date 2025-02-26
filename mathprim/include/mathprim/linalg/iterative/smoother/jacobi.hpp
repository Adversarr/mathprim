#pragma once
#include "smoother.hpp"
#include "mathprim/linalg/iterative/internal/diagonal_extract.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, sparse_format Format, typename Blas>
class jacobi_smoother : public basic_smoother<jacobi_smoother<Scalar, Device, Format, Blas>, Scalar, Device> {
  // D x' = b - (L + U) x.
  // x' = D^-1 (b - A x + D x) = D^-1 (b - A x) + x = D^-1 r + x.

  using base = basic_smoother<jacobi_smoother<Scalar, Device, Format, Blas>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = basic_sparse_view<const Scalar, Device, Format>;
  using buffer_type = contiguous_buffer<Scalar, shape_t<keep_dim>, Device>;
  friend base;

public:
  explicit jacobi_smoother(const_sparse mat) : base(mat.rows()), mat_(mat) {
    base::compute();
  }

  jacobi_smoother(jacobi_smoother&&) = default;

private:
  void compute_impl() {
    inv_diag_ = internal::diagonal_extract<Scalar, Device, Format>::extract(mat_);
  }

  void apply_impl(vector_type dx, const_vector residual) {
    blas_.copy(dx, residual); // dx = residual
    blas_.emul(inv_diag_.const_view(), dx); // dx = inv_diag_ * dx
  }

  buffer_type inv_diag_;
  const_sparse mat_;
  Blas blas_;
};

}  // namespace mathprim::sparse::iterative