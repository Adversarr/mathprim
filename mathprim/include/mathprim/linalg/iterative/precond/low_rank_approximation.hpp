#pragma once
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/core/utils/timed.hpp"
#include <cuda_runtime.h>
namespace mathprim::iterative_solver {

namespace internal {}

template <typename Scalar, typename Device, sparse::sparse_format Compression, typename Blas>
class low_rank_preconditioner
    : public basic_preconditioner<low_rank_preconditioner<Scalar, Device, Compression, Blas>, Scalar, Device> {
public:
  using const_sparse = sparse::basic_sparse_view<const Scalar, Device, Compression>;
  using buffer_type = continuous_buffer<Scalar, shape_t<keep_dim>, Device>;
  using base = basic_preconditioner<low_rank_preconditioner<Scalar, Device, Compression, Blas>, Scalar, Device>;
  friend base;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using basis_type = continuous_buffer<Scalar, dshape<2>, Device>;
  using diag_type = continuous_buffer<Scalar, dshape<1>, Device>;
  using temp_type = continuous_buffer<Scalar, dshape<1>, Device>;

  explicit low_rank_preconditioner(index_t n, index_t k) :
      mat_U_trans_(make_buffer<Scalar, Device>(make_shape(k, n))),
      diag_(make_buffer<Scalar, Device>(make_shape(k))),
      temp_(make_buffer<Scalar, Device>(make_shape(k))) {}

  low_rank_preconditioner(low_rank_preconditioner&&) = default;

  basis_type& basis() {
    return mat_U_trans_;
  }

  diag_type& diag() {
    return diag_;
  }

protected:
  // Y <- 1/s U * inv(D) * U^T * X + (I - U * U^T) * X
  // Y <- X + U * (inv(D)/s - I) *  U^T * X
  //              ^^^^^^^^^^^^^^ this part is the diag_ buffer.
  // Y <- X + U * diag_ * U^T * X
  void apply_impl(vector_type y, const_vector x) {
    auto u = mat_U_trans_.const_view().transpose();
    auto u_t = mat_U_trans_.const_view();

    auto subspace = temp_.view();

    // check for the size of the matrix.
    auto [n, k] = u.shape();
    auto n2 = x.shape(0);
    auto n3 = y.shape(0);
    MATHPRIM_INTERNAL_CHECK_THROW(n == n2 && n == n3, std::runtime_error,
                                  "The size of the matrix is not compatible with the preconditioner.");
    // Y <- X
    blas_.copy(y, x);

    // temp <- U^T * X
    blas_.gemv(1, u_t, x, 0, subspace);

    // Y <- D^-1 * temp
    blas_.emul(diag_.const_view(), subspace);

    // Y <- U * temp + X
    blas_.gemv(1, u, subspace.as_const(), 1, y);
  }

  // matrix U: low rank approximation for the inverse of the matrix.
  // matrix V: low rank approximation for the inverse of the matrix, optional for symmetric matrix.
  basis_type mat_U_trans_;  // k, n matrix, k is the rank of the approximation.
  diag_type diag_;    // k, eigen values. (inv(D) - 1)
  temp_type temp_;    // k, the intermediate buffer.
  Blas blas_;
};

}  // namespace mathprim::iterative_solver
