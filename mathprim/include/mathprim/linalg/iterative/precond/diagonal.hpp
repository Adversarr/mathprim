#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/linalg/iterative/internal/diagonal_extract.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, sparse::sparse_format Compression, typename Blas>
class diagonal_preconditioner : public basic_preconditioner<diagonal_preconditioner<Scalar, Device, Compression, Blas>,
                                                            Scalar, Device, Compression> {
public:
  using const_sparse = sparse::basic_sparse_view<const Scalar, Device, Compression>;
  using buffer_type = contiguous_buffer<Scalar, shape_t<keep_dim>, Device>;
  using this_type = diagonal_preconditioner<Scalar, Device, Compression, Blas>;
  using base = basic_preconditioner<this_type, Scalar, Device, Compression>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  friend base;

  diagonal_preconditioner() = default;
  explicit diagonal_preconditioner(const_sparse matrix) : base(matrix) { this->compute(); }
  diagonal_preconditioner(diagonal_preconditioner&&) = default;

protected:
  void factorize_impl() {
    auto spm = this->matrix();
    inv_diag_ = internal::diagonal_extract<Scalar, Device, Compression>::extract(spm);
  }

  // Y <- D^-1 * X
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(inv_diag_, std::runtime_error, "Preconditioner not initialized.");
    blas_.copy(y, x);  // Y = X
    blas_.emul(inv_diag_.const_view(), y);
  }

private:
  buffer_type inv_diag_;
  Blas blas_;
};

}  // namespace mathprim::sparse::iterative
