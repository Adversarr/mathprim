#pragma once
#include <limits>

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/view.hpp"
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
  using base = basic_linear_operator<sparse_matrix<SparseBlas>, typename SparseBlas::scalar_type,
                                     typename SparseBlas::device_type>;
  friend base;
  using Scalar = typename SparseBlas::scalar_type;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse_view = typename SparseBlas::const_sparse_view;

  explicit sparse_matrix(const_sparse_view mat) : base(mat.rows(), mat.cols()), spmv_(mat) {}

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
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;

  // y <- M^-1 * x
  void apply(vector_type y, const_vector x) {
    static_cast<Derived*>(this)->apply_impl(y, x);
  }
};

template <typename Scalar, typename Device>
class none_preconditioner : public basic_preconditioner<none_preconditioner<Scalar, Device>, Scalar, Device> {
public:
  using base = basic_preconditioner<none_preconditioner<Scalar, Device>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;

  none_preconditioner() = default;
  none_preconditioner(none_preconditioner&&) = default;

  void apply_impl(vector_type y, const_vector x) {
    ::mathprim::copy(y, x);  // Y <- X
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Iterative Solver Base
///////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
struct iterative_solver_parameters {
  index_t max_iterations_;
  Scalar norm_tol_;
};

template <typename Scalar>
struct iterative_solver_result {
  index_t iterations_ = {1 << 10};                         ///< number of iterations.
  Scalar norm_ = {std::numeric_limits<float>::epsilon()};  ///< l2 norm of the residual. (norm(r))
};

template <typename Derived, typename Scalar, typename Device, typename LinearOperatorT, typename BlasT,
          typename PreconditionerT = none_preconditioner<Scalar, Device>>
class basic_iterative_solver {
public:
  using scalar_type = Scalar;
  using linear_operator_type = LinearOperatorT;
  using blas_type = BlasT;
  using preconditioner_type = PreconditionerT;
  using results_type = iterative_solver_result<Scalar>;
  using parameters_type = iterative_solver_parameters<Scalar>;
  using vector_type = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;

  explicit basic_iterative_solver(linear_operator_type matrix, blas_type blas = {},
                                  preconditioner_type preconditioner = {}) :
      matrix_(std::move(matrix)),
      blas_(std::move(blas)),
      preconditioner_(std::move(preconditioner)),
      residual_(make_buffer<Scalar, Device>(make_shape(matrix_.rows()))) {}

  // NOT necessary since we use CRTP.
  // virtual ~basic_iterative_solver() = default;
  struct no_op {
    inline void operator()(index_t /* iter */, Scalar /* norm */) const noexcept {}
  };

  // Solve the linear system.
  template <typename Callback = no_op>
  MATHPRIM_NOINLINE results_type apply(const_vector b, vector_type x, const parameters_type& params = {},
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
    return static_cast<Derived*>(this)->template apply_impl<Callback>(b, x, params, std::forward<Callback>(cb));
  }

  const_vector residual() const noexcept {
    return residual_.view();
  }

  linear_operator_type& linear_operator() noexcept {
    return matrix_;
  }

  blas_type& blas() noexcept {
    return blas_;
  }

  preconditioner_type& preconditioner() noexcept {
    return preconditioner_;
  }

protected:
  linear_operator_type matrix_;
  blas_type blas_;
  preconditioner_type preconditioner_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> residual_;
};

}  // namespace mathprim::sparse::iterative