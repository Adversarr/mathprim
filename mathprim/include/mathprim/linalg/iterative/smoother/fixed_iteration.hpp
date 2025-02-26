#pragma once
#include "smoother.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, sparse_format Format, typename LinearOperator, typename Blas,
          typename Smoother>
class basic_fixed_iteration
    : public basic_iterative_solver<basic_fixed_iteration<Scalar, Device, Format, LinearOperator, Blas, Smoother>,
                                    Scalar, Device, LinearOperator> {
private:
  using base = basic_iterative_solver<basic_fixed_iteration<Scalar, Device, Format, LinearOperator, Blas, Smoother>,
                                      Scalar, Device, LinearOperator>;
  friend base;

  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using linear_operator_type = typename base::linear_operator_type;
  using results_type = typename base::results_type;
  using parameters_type = typename base::parameters_type;

public:
  basic_fixed_iteration(linear_operator_type matrix, Blas blas, Smoother smoother) :
      base(std::move(matrix)), blas_(std::move(blas)), smoother_(std::move(smoother)) {}

private:
  template <typename Callback>
  results_type apply_impl(const_vector b, vector_type x, const parameters_type& params, Callback&& callback) {
    auto& blas = blas_;
    auto& smoother = smoother_;
    auto& matrix = base::matrix_;
    auto& residual_buffer = base::residual_;
    if (!dx_ || dx_.size() != x.size()) {
      dx_ = make_buffer<Scalar, Device>(x.shape());
    }

    vector_type r = residual_buffer.view(), dx = dx_.view();
    const_vector cx = x.as_const(), cr = r.as_const(), cdx = dx_.view();
    const Scalar b_norm = blas.norm(b);

    // initialize r <- b - A * x
    blas.copy(r, b);
    matrix.apply(-1, cx, 1, r);

    // initialize results
    results_type results;
    auto& iterations = results.iterations_;
    auto& norm = results.norm_;
    iterations = 0;
    norm = blas.norm(cr) / b_norm;

    // iterate
    for (; iterations < params.max_iterations_; ++iterations) {
      // 1. smoother
      smoother.apply(dx, r); // solves M * dx = r
      // 2. update x and residual
      blas.axpy(1, dx, x);   // x <- x + (delta x)
      matrix.apply(-1, dx, 1, r);  // r <- r - A * (delta x)

      // check convergence
      norm = blas.norm(cr) / b_norm;
      callback(iterations, norm);
      if (norm < params.norm_tol_) {
        break;
      }
    }

    // set results
    results.iterations_ = iterations;
    results.norm_ = norm;
    return results;
  }
  Blas blas_;
  Smoother smoother_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> dx_;
};
}  // namespace mathprim::sparse::iterative