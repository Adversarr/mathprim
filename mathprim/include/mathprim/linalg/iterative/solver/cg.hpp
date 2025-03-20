#pragma once
#include <cmath>

#include "mathprim/blas/blas.hpp" // IWYU pragma: export
#include "mathprim/linalg/iterative/iterative.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, typename SparseBlas, typename BlasT,
          typename PreconditionerT = none_preconditioner<Scalar, Device, SparseBlas::compression>>
class cg : public basic_iterative_solver<cg<Scalar, Device, SparseBlas, BlasT, PreconditionerT>, Scalar, Device,
                                         SparseBlas> {
public:
  using this_type = cg<Scalar, Device, SparseBlas, BlasT, PreconditionerT>;
  using base = basic_iterative_solver<this_type, Scalar, Device, SparseBlas>;
  friend base;
  static constexpr sparse_format compression = SparseBlas::compression;
  using base2 = basic_sparse_solver<this_type, Scalar, Device, SparseBlas::compression>;
  friend base2;

  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using sparse_view = typename base::sparse_view;
  using const_sparse = typename base::const_sparse;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  using results_type = convergence_result<Scalar>;
  using parameters_type = convergence_criteria<Scalar>;
  using scalar_type = Scalar;

  using blas_type = BlasT;
  using preconditioner_type = PreconditionerT;

  using precond_ref = basic_preconditioner<preconditioner_type, Scalar, Device, compression>&;
  using precond_const_ref = const basic_preconditioner<preconditioner_type, Scalar, Device, compression>&;

  cg() = default;
  explicit cg(const_sparse matrix) : base(matrix) { this->compute({}); }
  MATHPRIM_INTERNAL_MOVE(cg, default);
  MATHPRIM_INTERNAL_COPY(cg, delete);

  blas_type& blas() noexcept { return blas_; }

  precond_ref preconditioner() noexcept { return preconditioner_; }
  precond_const_ref preconditioner() const noexcept { return preconditioner_; }

  template <typename Callback>
  results_type solve_impl(vector_view x, const_vector b, const parameters_type& params, Callback&& cb) {
    mathprim::blas::basic_blas<blas_type, Scalar, Device>& blas = blas_;
    auto& preconditioner = preconditioner_;
    auto& matrix = this->linear_operator();
    auto& residual_buffer = base::residual_;
    vector_view r = residual_buffer.view(), q = q_.view(), d = d_.view();
    const Scalar b_norm = blas.norm(b);
    MATHPRIM_INTERNAL_CHECK_THROW(b_norm > 0, std::runtime_error,
                                  "Norm of b is invalid, |b|=" + std::to_string(b_norm));

    // r = b - A * x
    blas.copy(r, b);           // r = b
    matrix.gemv(-1, x, 1, r);  // r = b - A * x

    // Initialize.
    results_type results;
    auto& iterations = results.iterations_;
    auto& norm = results.norm_;
    // auto& max_norm = results.amax_;
    iterations = 0;
    norm = blas.norm(r) / b_norm;
    // max_norm = blas.amax(cr);
    bool converged = norm <= params.norm_tol_;
    // Set initial search direction.
    preconditioner.apply(d, r);           // d = M^-1 * r
    Scalar delta_new = blas.dot(r, d);  // delta_new = (r, d)

    // Main loop.
    for (; iterations < params.max_iterations_; ++iterations) {
      // q = A * d
      matrix.gemv(1, d, 0, q);  // q = A * d

      // alpha = (r, d) / (d, q)
      Scalar d_q = blas.dot(d, q);
      Scalar alpha = delta_new / d_q;

      // x = x + alpha * d
      blas.axpy(alpha, d, x);  // x = x + alpha * d

      // r = r - alpha * q
      // if ((1 + iterations) % 50 == 0) {
      //   blas.copy(r, b);           // r = b
      //   matrix.gemv(-1, x, 1, r);  // r = b - A * x
      // } else {
      //   blas.axpy(-alpha, q, r);  // r = r - alpha * q
      // }
      blas.axpy(-alpha, q, r);  // r = r - alpha * q

      // Check convergence.
      norm = blas.norm(r) / b_norm;
      MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(norm), std::runtime_error, "Norm is not finite.");
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
      delta_new = blas.dot(r, q);           // delta_new = (r, q)
      Scalar beta = delta_new / delta_old;  // beta = delta_new / delta_old
      // update the search direction: d = q + beta * d
      blas.axpy(beta, d, q);  // q = q + beta * d
      d.swap(q);
    }

    return results;
  }

protected:
  void analyze_impl_impl() {
    auto mat = this->matrix();
    preconditioner().analyze(mat);
    q_ = make_buffer<scalar_type, Device>(mat.rows());
    d_ = make_buffer<scalar_type, Device>(mat.rows());
  }

  void factorize_impl_impl() { preconditioner_.factorize(); }

  blas_type blas_;
  preconditioner_type preconditioner_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> q_;  // temporary buffer
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> d_;  // search direction
};

}  // namespace mathprim::sparse::iterative
