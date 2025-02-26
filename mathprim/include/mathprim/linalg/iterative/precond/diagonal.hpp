#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/linalg/iterative/internal/diagonal_extract.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
namespace mathprim::sparse::iterative {


template <typename Scalar, typename Device, sparse::sparse_format Compression, typename Blas>
class diagonal_preconditioner
    : public basic_preconditioner<diagonal_preconditioner<Scalar, Device, Compression, Blas>, Scalar, Device> {
public:
  using const_sparse = sparse::basic_sparse_view<const Scalar, Device, Compression>;
  using buffer_type = contiguous_buffer<Scalar, shape_t<keep_dim>, Device>;
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

}  // namespace mathprim::sparse::iterative

