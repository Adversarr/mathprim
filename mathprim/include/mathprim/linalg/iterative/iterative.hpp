#pragma once
#include <limits>

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/view.hpp"
#include "mathprim/linalg/basic_sparse_solver.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse::iterative {

///////////////////////////////////////////////////////////////////////////////
/// Linear Operator
///////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename Scalar, typename Device>
class basic_linear_operator {
public:
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;

  index_t rows() const {
    return rows_;
  }
  index_t cols() const {
    return cols_;
  }

  basic_linear_operator(index_t rows, index_t cols) : rows_(rows), cols_(cols) {}

  // y <- alpha * A * x + beta * y
  void apply(Scalar alpha, const_vector x, Scalar beta, vector_type y) {
    static_cast<Derived*>(this)->apply_impl(alpha, x, beta, y);
  }

  // y <- alpha * A.T * x + beta * y
  void apply_transpose(Scalar alpha, const_vector x, Scalar beta, vector_type y) {
    static_cast<Derived*>(this)->apply_transpose_impl(alpha, x, beta, y);
  }

private:
  index_t rows_;
  index_t cols_;
};

template <typename SparseBlas>
class sparse_matrix : public basic_linear_operator<sparse_matrix<SparseBlas>, typename SparseBlas::scalar_type,
                                                   typename SparseBlas::device_type> {
public:
  static constexpr sparse_format compression = SparseBlas::compression;
  using base = basic_linear_operator<sparse_matrix<SparseBlas>, typename SparseBlas::scalar_type,
                                     typename SparseBlas::device_type>;
  friend base;
  using Scalar = typename SparseBlas::scalar_type;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse_view = typename SparseBlas::const_sparse_view;

  explicit sparse_matrix(const_sparse_view mat) : base(mat.rows(), mat.cols()), spmv_(mat) {}

  const_sparse_view matrix() const noexcept { return spmv_.matrix(); }

protected:
  void apply_impl(Scalar alpha, const_vector x, Scalar beta, vector_type y) {
    spmv_.gemv(alpha, x, beta, y, false);
  }

  void apply_transpose_impl(Scalar alpha, const_vector x, Scalar beta, vector_type y) {
    if (spmv_.matrix().property() == sparse::sparse_property::symmetric) {
      spmv_.gemv(alpha, x, beta, y, true);
    } else {
      throw std::runtime_error("Not implemented for unsymmetric matrix.");
    }
  }


private:
  SparseBlas spmv_;
};

///////////////////////////////////////////////////////////////////////////////
/// Preconditioner
///////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename Scalar, typename Device>
class basic_preconditioner {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;

  // y <- M^-1 * x
  void apply(vector_type y, const_vector x) {
    static_cast<Derived*>(this)->apply_impl(y, x);
  }

  // TODO: Implement this logic.
  template <typename SparseMatrixT>
  void analyze(const SparseMatrixT& matrix) {
    static_cast<Derived*>(this)->analyze_impl(matrix);
  }

  template <typename SparseMatrixT>
  void factorize(const SparseMatrixT& matrix) {
    static_cast<Derived*>(this)->factorize_impl(matrix);
  }

  template <typename SparseMatrixT>
  void analyze_impl(const SparseMatrixT& /* matrix */) {}

  template <typename SparseMatrixT>
  void factorize_impl(const SparseMatrixT& /* matrix */) {}
};

template <typename Scalar, typename Device>
class none_preconditioner : public basic_preconditioner<none_preconditioner<Scalar, Device>, Scalar, Device> {
public:
  using base = basic_preconditioner<none_preconditioner<Scalar, Device>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;

  none_preconditioner() = default;
  template <typename SparseMatrixT>
  none_preconditioner(const SparseMatrixT& /* matrix */) {}  // NOLINT(google-explicit-constructor)
  none_preconditioner(none_preconditioner&&) = default;

  void apply_impl(vector_type y, const_vector x) {
    ::mathprim::copy(y, x);  // Y <- X
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Iterative Solver Base
///////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename Scalar, typename Device, typename LinearOperatorT>
class basic_iterative_solver {
public:
  using scalar_type = Scalar;
  using linear_operator_type = LinearOperatorT;
  using results_type = convergence_result<Scalar>;
  using parameters_type = convergence_criteria<Scalar>;
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;

  explicit basic_iterative_solver(linear_operator_type matrix) :
      matrix_(std::move(matrix)),
      residual_(make_buffer<Scalar, Device>(make_shape(matrix_.rows()))) {}

  // NOT necessary since we use CRTP.
  // virtual ~basic_iterative_solver() = default;
  struct no_op {
    inline void operator()(index_t /* iter */, Scalar /* norm */) const noexcept {}
  };

  // Solve the linear system.
  template <typename Callback = no_op>
  MATHPRIM_NOINLINE results_type solve(vector_type x, const_vector b, const parameters_type& params = {},
                                       Callback&& cb = {}) {
    // 1. Check the size of b and x.
    const index_t b_size = b.size();
    const index_t x_size = x.size();
    if (b_size != matrix_.rows()) {
      throw std::runtime_error("The size of b is not equal to the number of rows of the matrix.");
    }
    if (x_size != matrix_.cols()) {
      throw std::runtime_error("The size of x is not equal to the number of cols of the matrix.");
    }

    // 2. Check parameter values.
    if (params.norm_tol_ <= 0 /* || params.amax_tol_ <= 0 */) {
      throw std::runtime_error("The tolerance must be positive.");
    }
    if (params.max_iterations_ <= 0) {
      throw std::runtime_error("The maximum number of iterations must be positive.");
    }

    // 3. Apply the solver.
    return static_cast<Derived*>(this)->template solve_impl<Callback>(x, b, params, std::forward<Callback>(cb));
  }

  const_vector residual() const noexcept {
    return residual_.view();
  }

  linear_operator_type& linear_operator() noexcept {
    return matrix_;
  }

protected:
  linear_operator_type matrix_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> residual_;
};

///////////////////////////////////////////////////////////////////////////////
/// Specialization for LinearOperator == SparseMatrix
///////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename Scalar, typename Device, typename SparseBlas>
class basic_iterative_solver<Derived, Scalar, Device, sparse_matrix<SparseBlas>>
    : public basic_sparse_solver<Derived, Scalar, Device, sparse_matrix<SparseBlas>::compression> {
public:
  using base = basic_sparse_solver<Derived, Scalar, Device, sparse_matrix<SparseBlas>::compression>;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using sparse_view = typename base::sparse_view;
  using const_sparse = typename base::const_sparse;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  using results_type = convergence_result<Scalar>;
  using parameters_type = convergence_criteria<Scalar>;
  using linear_operator_type = sparse_matrix<SparseBlas>;
  friend base;

  explicit basic_iterative_solver(linear_operator_type matrix) : base(matrix.matrix()), matrix_(std::move(matrix)) {
    residual_ = make_buffer<Scalar, Device>(make_shape(matrix_.rows()));
  }
  basic_iterative_solver(basic_iterative_solver&&) = default;
  basic_iterative_solver& operator=(basic_iterative_solver&&) = default;
  void vsolve_impl(matrix_view /* lhs */, const_matrix_view /* rhs */, const parameters_type& /* params */) {
    throw std::runtime_error("Iterative solver does not support vectorized solve.");
  }
  const_vector residual() const noexcept { return residual_.view(); }
  linear_operator_type& linear_operator() noexcept { return matrix_; }

protected:
  linear_operator_type matrix_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> residual_;
};

}  // namespace mathprim::sparse::iterative