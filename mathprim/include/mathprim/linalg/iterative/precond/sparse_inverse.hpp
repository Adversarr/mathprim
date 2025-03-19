#pragma once
#include <cmath>
#include "mathprim/core/buffer.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"

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

  void set_approximation(const_sparse mat, Scalar eps) {
    bl_ = SparseBlas(mat);
    buffer_intern_ = make_buffer<Scalar, Device>(mat.rows());
    eps_ = eps;
  }

private:
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(buffer_intern_, std::runtime_error, "Preconditioner not initialized.");
    // z = lo.T * x.
    auto z = buffer_intern_.view();
    bl_.gemv(1, x, 0, z, true);
    // y = lo * y.
    bl_.gemv(1, z, 0, y, false);
    // y = y + eps x
    dense_bl_.axpy(eps_, x, y);
  }

  SparseBlas bl_;
  Scalar eps_;
  Blas dense_bl_;
  contiguous_vector_buffer<Scalar, Device> buffer_intern_;
};
}  // namespace mathprim::sparse::iterative