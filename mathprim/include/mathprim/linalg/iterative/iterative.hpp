#pragma once
#include <stdexcept>

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/view.hpp"
#include "mathprim/linalg/basic_sparse_solver.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse::iterative {

///////////////////////////////////////////////////////////////////////////////
/// Linear Operator
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Preconditioner
///////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename Scalar, typename Device, sparse_format Compression>
class basic_preconditioner {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;
  using sparse_view = basic_sparse_view<Scalar, Device, Compression>;
  using const_sparse = basic_sparse_view<const Scalar, Device, Compression>;

  basic_preconditioner() = default;
  explicit basic_preconditioner(const_sparse mat) noexcept : matrix_(mat) {}

  Derived& derived() noexcept { return *static_cast<Derived*>(this); }
  const Derived& derived() const noexcept { return *static_cast<const Derived*>(this); }

  // y <- M^-1 * x
  void apply(vector_type y, const_vector x) { static_cast<Derived*>(this)->apply_impl(y, x); }

  // TODO: Implement this logic.
  Derived& analyze(const_sparse matrix) {
    if (matrix) {
      matrix_ = matrix;
    }
    MATHPRIM_INTERNAL_CHECK_THROW(matrix_, std::runtime_error, "No matrix provided.");
    derived().analyze_impl();
    return derived();
  }

  Derived& factorize() {
    MATHPRIM_INTERNAL_CHECK_THROW(matrix_, std::runtime_error, "No matrix provided.");
    derived().factorize_impl();
    return derived();
  }

  Derived& compute(const_sparse matrix) { return analyze(matrix).factorize(); }

  void analyze_impl() const noexcept {}
  void factorize_impl() const noexcept {}

  const_sparse matrix() const noexcept { return matrix_; }

protected:
  const_sparse matrix_;
};

template <typename Scalar, typename Device, sparse_format Compression>
class none_preconditioner final
    : public basic_preconditioner<none_preconditioner<Scalar, Device, Compression>, Scalar, Device, Compression> {
public:
  using base = basic_preconditioner<none_preconditioner<Scalar, Device, Compression>, Scalar, Device, Compression>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using sparse_view = typename base::sparse_view;
  using const_sparse = typename base::const_sparse;
  friend base;

  none_preconditioner() = default;
  explicit none_preconditioner(const_sparse mat) : base(mat) { this->compute({}); }
  none_preconditioner(none_preconditioner&&) = default;

protected:
  void apply_impl(vector_type y, const_vector x) {
    ::mathprim::copy(y, x);  // Y <- X
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Iterative Solver Base
///////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename Scalar, typename Device, typename SparseBlas>
class basic_iterative_solver : public basic_sparse_solver<Derived, Scalar, Device, SparseBlas::compression> {
public:
  using base = basic_sparse_solver<Derived, Scalar, Device, SparseBlas::compression>;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using sparse_view = typename base::sparse_view;
  using const_sparse = typename base::const_sparse;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  using results_type = convergence_result<Scalar>;
  using parameters_type = convergence_criteria<Scalar>;
  friend base;
  basic_iterative_solver() = default;
  explicit basic_iterative_solver(const_sparse matrix) : base(matrix), sp_blas_(matrix) {
    residual_ = make_buffer<Scalar, Device>(matrix.rows());
  }
  MATHPRIM_INTERNAL_MOVE(basic_iterative_solver, default);
  MATHPRIM_INTERNAL_COPY(basic_iterative_solver, delete);

  void vsolve_impl(matrix_view /* lhs */, const_matrix_view /* rhs */, const parameters_type& /* params */) {
    throw std::runtime_error("Iterative solver does not support vectorized solve.");
  }

  const_vector residual() const noexcept { return residual_.view(); }

  void analyze_impl() {
    residual_ = make_buffer<Scalar, Device>(this->matrix().rows());
    // ugly.
    static_cast<Derived*>(this)->analyze_impl_impl();
  }

  void factorize_impl() { static_cast<Derived*>(this)->factorize_impl_impl(); }

  SparseBlas& linear_operator() noexcept { return sp_blas_; }
  const SparseBlas& linear_operator() const noexcept { return sp_blas_; }

protected:
  SparseBlas sp_blas_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> residual_;
};

}  // namespace mathprim::sparse::iterative
