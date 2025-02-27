#pragma once
#include "mathprim/linalg/iterative/iterative.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, typename LinearOperatorT, typename BlasT,
          typename PreconditionerT = none_preconditioner<Scalar, Device>>
class cg : public basic_iterative_solver<cg<Scalar, Device, LinearOperatorT, BlasT, PreconditionerT>, Scalar, Device,
                                         LinearOperatorT> {
public:
  using base = basic_iterative_solver<cg<Scalar, Device, LinearOperatorT, BlasT, PreconditionerT>, Scalar, Device,
                                      LinearOperatorT>;
  using scalar_type = typename base::scalar_type;
  using linear_operator_type = typename base::linear_operator_type;
  using results_type = typename base::results_type;
  using parameters_type = typename base::parameters_type;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using blas_type = BlasT;
  using preconditioner_type = PreconditionerT;

  explicit cg(linear_operator_type matrix, blas_type blas = {}, preconditioner_type preconditioner = {}) :
      base(std::move(matrix)),
      blas_(std::move(blas)),
      preconditioner_(std::move(preconditioner)),
      q_(make_buffer<scalar_type, Device>(make_shape(base::matrix_.rows()))),
      d_(make_buffer<scalar_type, Device>(make_shape(base::matrix_.rows()))) {}

  blas_type& blas() noexcept {
    return blas_;
  }

  preconditioner_type& preconditioner() noexcept {
    return preconditioner_;
  }

  template <typename Callback>
  results_type apply_impl(const_vector b, vector_type x, const parameters_type& params, Callback&& cb) {
    auto& blas = blas_;
    auto& preconditioner = preconditioner_;
    auto& matrix = base::matrix_;
    auto& residual_buffer = base::residual_;
    const_vector cx = x.as_const();
    vector_type r = residual_buffer.view(), q = q_.view(), d = d_.view();
    const_vector cr = r.as_const(), cq = q.as_const(), cd = d.as_const();
    const Scalar b_norm = blas.norm(b);

    // r = b - A * x
    blas.copy(r, b);             // r = b
    matrix.apply(-1, cx, 1, r);  // r = b - A * x

    // Initialize.
    results_type results;
    auto& iterations = results.iterations_;
    auto& norm = results.norm_;
    // auto& max_norm = results.amax_;
    iterations = 0;
    norm = blas.norm(cr) / b_norm;
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
      norm = blas.norm(cr) / b_norm;
      // max_norm = blas.amax(cr);
      converged = norm <= params.norm_tol_ /* || max_norm <= params.amax_tol_ */;
      cb(iterations, norm);
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

    norm /= b_norm;
    return results;
  }

private:
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> q_;  // temporary buffer
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> d_;  // search direction
  blas_type blas_;
  preconditioner_type preconditioner_;
};

}  // namespace mathprim::sparse::iterative
