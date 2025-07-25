#pragma once
#include <cmath>

#include "mathprim/blas/blas.hpp"  // IWYU pragma: export
#include "mathprim/core/defines.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, typename SparseBlas, typename BlasT,
          typename PreconditionerT = none_preconditioner<Scalar, Device, SparseBlas::compression>>
class cg : public basic_iterative_solver<cg<Scalar, Device, SparseBlas, BlasT, PreconditionerT>, Scalar, Device,
                                         SparseBlas> {
public:
  using self_type = cg<Scalar, Device, SparseBlas, BlasT, PreconditionerT>;
  using base = basic_iterative_solver<self_type, Scalar, Device, SparseBlas>;
  friend base;
  static constexpr sparse_format compression = SparseBlas::compression;
  using base2 = basic_sparse_solver<self_type, Scalar, Device, SparseBlas::compression>;
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
    auto& preconditioner = this->preconditioner();
    auto& matrix = this->linear_operator();
    auto& residual_buffer = base::residual_;
    vector_view r = residual_buffer.view(), q = q_.view(), d = d_.view();
    zeros(r);
    zeros(q);
    zeros(d);

    const Scalar b_norm = blas.norm(b);
    MATHPRIM_INTERNAL_CHECK_THROW(b_norm > 0, std::runtime_error,
                                  "Norm of b is invalid, |b|=" + std::to_string(b_norm));
    const Scalar x0_norm = blas.norm(x);
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(x0_norm), std::runtime_error,
                                  "Norm of x is invalid, |x|=" + std::to_string(x0_norm));

    // Initialize.
    results_type results;
    auto& iterations = results.iterations_;
    auto& norm = results.norm_;
    iterations = 0;
    // r = b - A * x
    copy(r, b);                // r = b
    matrix.gemv(-1, x, 1, r);  // r = b - A * x
    // Set initial search direction.
    preconditioner.apply(d, r);         // d = M^-1 * r
    Scalar delta_new = blas.dot(r, d);  // delta_new = (r, d)
    bool converged = false;
    // Check convergence.
    norm = blas.norm(r) / b_norm;
#define MATHPRIM_INTERNAL_CG_CHECK(x) \
  do {                                \
    if (!std::isfinite(x))            \
      throw_traced(#x "not finite");  \
  } while (0)
    if (converged) {
      return results;
    }
    Scalar d_q = 0, alpha = 0;
    auto throw_traced = [&](const char* msg) -> void {
      throw std::runtime_error(                                   //
          "check \"" + std::string(msg) +                         //
          "\" failed at iteration " + std::to_string(iterations)  //
          + ", |r|=" + std::to_string(norm)                       //
          + ", |x|=" + std::to_string(blas.norm(x))               //
          + ", |x0|=" + std::to_string(x0_norm)                   //
          + ", |b|=" + std::to_string(b_norm)                     //
          + ", |d|=" + std::to_string(blas.norm(d))               //
          + ", |q|=" + std::to_string(blas.norm(q))               //
          + ", <d,q>=" + std::to_string(d_q)                      //
          + ", alpha=" + std::to_string(alpha)                    //
          + ", delta=" + std::to_string(delta_new)                //
      );
    };
    if (!(delta_new > 0)) {
      throw_traced("!(delta_new > 0)");
    }
    MATHPRIM_INTERNAL_CG_CHECK(norm);

    for (; iterations < params.max_iterations_; ++iterations) {
      // q = A * d
      matrix.gemv(1, d, 0, q);  // q = A * d

      // alpha = (r, d) / (d, q)
      d_q = blas.dot(d, q);
      MATHPRIM_INTERNAL_CG_CHECK(d_q);
      alpha = delta_new / d_q;
      MATHPRIM_INTERNAL_CG_CHECK(alpha);

      // x = x + alpha * d
      blas.axpy(alpha, d, x);  // x = x + alpha * d

      // r = r - alpha * q
      if constexpr (std::is_same_v<Scalar, double>) {
        if ((1 + iterations) % 50 == 0) {
          blas.copy(r, b);           // r = b
          matrix.gemv(-1, x, 1, r);  // r = b - A * x
        } else {
          blas.axpy(-alpha, q, r);  // r = r - alpha * q
        }
      } else {
        blas.axpy(-alpha, q, r);  // r = r - alpha * q
      }

      // Check convergence.
      norm = blas.norm(r) / b_norm;
      converged = norm <= params.norm_tol_;
      cb(iterations, norm);
      if (converged) {
        break;
      }

      // Update search direction.
      const Scalar delta_old = delta_new;  // delta_old = delta_new
      // q is not needed anymore, so we can use it as a temporary buffer.
      preconditioner.apply(q, r);           // q = M^-1 * r
      delta_new = blas.dot(r, q);           // delta_new = (r, q)
      Scalar beta = delta_new / delta_old;  // beta = delta_new / delta_old
      MATHPRIM_INTERNAL_CG_CHECK(beta);

      // update the search direction: d = q + beta * d
      blas.axpy(beta, d, q);  // q = q + beta * d
      d.swap(q);
    }
#undef MATHPRIM_INTERNAL_CG_CHECK

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

  index_t cg_restart_threshold_ = 50;  ///< Maximum number of CG restart trials.
  blas_type blas_;
  preconditioner_type preconditioner_;
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> q_;  // temporary buffer
  contiguous_buffer<Scalar, shape_t<keep_dim>, Device> d_;  // search direction
};

}  // namespace mathprim::sparse::iterative
