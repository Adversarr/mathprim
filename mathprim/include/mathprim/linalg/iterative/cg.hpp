#pragma once
#include <iostream>

#include "mathprim/linalg/iterative/iterative.hpp"
namespace mathprim::iterative_solver {

template <typename Scalar, typename device, typename LinearOperatorT, typename BlasT,
          typename PreconditionerT = none_preconditioner<Scalar, device>>
class cg : public basic_iterative_solver<cg<Scalar, device, LinearOperatorT, BlasT, PreconditionerT>, Scalar, device,
                                         LinearOperatorT, BlasT, PreconditionerT> {
public:
  using base = basic_iterative_solver<cg<Scalar, device, LinearOperatorT, BlasT, PreconditionerT>, Scalar, device,
                                      LinearOperatorT, BlasT, PreconditionerT>;
  using scalar_type = typename base::scalar_type;
  using linear_operator_type = typename base::linear_operator_type;
  using blas_type = typename base::blas_type;
  using preconditioner_type = typename base::preconditioner_type;
  using results_type = typename base::results_type;
  using parameters_type = typename base::parameters_type;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;

  explicit cg(linear_operator_type matrix, blas_type blas = {}, preconditioner_type preconditioner = {}) :
      base(std::move(matrix), std::move(blas), std::move(preconditioner)),
      q_(make_buffer<scalar_type, device>(make_shape(base::matrix_.rows()))),
      d_(make_buffer<scalar_type, device>(make_shape(base::matrix_.rows()))) {}

  results_type apply_impl(const_vector b, vector_type x, const parameters_type& params) {
    auto& blas = base::blas_;
    auto& matrix = base::matrix_;
    auto& preconditioner = base::preconditioner_;
    auto& residual_buffer = base::residual_;
    const_vector cx = x.as_const();
    vector_type r = residual_buffer.view(), q = q_.view(), d = d_.view();
    const_vector cr = r.as_const(), cq = q.as_const(), cd = d.as_const();

    // r = b - A * x
    blas.copy(r, b);             // r = b
    matrix.apply(-1, cx, 1, r);  // r = b - A * x

    // Initialize.
    results_type results;
    auto& iterations = results.iterations_;
    auto& norm = results.norm_;
    // auto& max_norm = results.amax_;
    iterations = 0;
    norm = blas.norm(cr);
    // max_norm = blas.amax(cr);
    bool converged = norm <= params.norm_tol_ /* || max_norm <= params.amax_tol_ */;
    // Set initial search direction.
    preconditioner.apply(d, r);           // d = M^-1 * r
    Scalar delta_new = blas.dot(cr, cd);  // delta_new = (r, d)

    // Main loop.
    for (; iterations < params.max_iterations_; ++iterations) {
      // q = A * d
      matrix.apply(1, cd, 0, q);  // q = A * d

      // alpha = (r, d) / (d, q)
      Scalar d_q = blas.dot(cd, cq);
      Scalar alpha = delta_new / d_q;

      // x = x + alpha * d
      blas.axpy(alpha, cd, x);  // x = x + alpha * d

      // r = r - alpha * q
      blas.axpy(-alpha, cq, r);  // r = r - alpha * q

      // Check convergence.
      norm = blas.norm(cr);
      // max_norm = blas.amax(cr);
      converged = norm <= params.norm_tol_ /* || max_norm <= params.amax_tol_ */;
      if (converged) {
        break;
      }

      // Update search direction.
      Scalar delta_old = delta_new;  // delta_old = delta_new
      // q is not needed anymore, so we can use it as a temporary buffer.
      preconditioner.apply(q, r);           // q = M^-1 * r
      delta_new = blas.dot(cr, cq);         // delta_new = (r, q)
      Scalar beta = delta_new / delta_old;  // beta = delta_new / delta_old
      // update the search direction: d = q + beta * d
      blas.axpy(beta, cd, q);  // q = q + beta * d
      d.swap(q);
      cd.swap(cq);
    }

    return results;
  }

private:
  continuous_buffer<Scalar, shape_t<keep_dim>, device> q_;  // temporary buffer
  continuous_buffer<Scalar, shape_t<keep_dim>, device> d_;  // search direction
};
}  // namespace mathprim::iterative_solver
