#pragma once
#include "mathprim/linalg/direct/eigen_support.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/linesearcher/backtracking.hpp"
namespace mathprim::optim {

template <typename Scalar, typename Device, typename Blas,
          typename Linesearcher = backtracking_linesearcher<Scalar, Device, Blas>,
          typename SparseSolver = sparse::direct::eigen_simplicial_ldlt<Scalar, sparse::sparse_format::csr>>
class newton_optimizer
    : public basic_optimizer<newton_optimizer<Scalar, Device, Blas, Linesearcher, SparseSolver>, Scalar, Device> {
public:
  using base = basic_optimizer<newton_optimizer<Scalar, Device, Blas, Linesearcher, SparseSolver>, Scalar, Device>;
  friend base;

  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;

  static constexpr sparse::sparse_format compression = SparseSolver::compression;

  using solver_ref_t = sparse::basic_sparse_solver<SparseSolver, Scalar, Device, compression>&;
  using solver_const_ref_t = const sparse::basic_sparse_solver<SparseSolver, Scalar, Device, compression>&;
  using blas_ref_t = blas::basic_blas<Blas, Scalar, Device>&;
  using blas_const_ref_t = const blas::basic_blas<Blas, Scalar, Device>&;
  using linesearch_ref_t = basic_linesearcher<Linesearcher, Scalar, Device>&;
  using linesearch_const_ref_t = const basic_linesearcher<Linesearcher, Scalar, Device>&;

  using sparse_view = sparse::basic_sparse_view<Scalar, Device, compression>;
  using const_sparse = sparse::basic_sparse_view<const Scalar, Device, compression>;
  using hessian_fn = std::function<std::pair<bool, const_sparse>()>;

  newton_optimizer() = default;
  MATHPRIM_INTERNAL_MOVE(newton_optimizer, default);
  MATHPRIM_INTERNAL_COPY(newton_optimizer, delete);

  newton_optimizer& set_hessian_fn(hessian_fn update_hessian) {
    update_hessian_ = update_hessian;
    return *this;
  }

  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback) {
    blas_ref_t blas = blas_;
    linesearch_ref_t ls = linesearcher_;
    solver_ref_t solver = solver_;

    auto criteria = this->criteria();
    auto grads = problem.fused_gradients().as_const();
    result_type result;
    Scalar& value = result.value_;
    Scalar& last_change = result.last_change_;
    Scalar& grad_norm = result.grad_norm_;
    index_t& iteration = result.iterations_;
    bool& converged = result.converged_;

    value = problem.eval_value_and_gradients();
    grad_norm = blas.norm(grads);
    converged = grad_norm < criteria.tol_grad_;
    if (converged) {
      // lucky path.
      return result;
    }
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(value), std::runtime_error, "Initial value is not finite.");
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(grad_norm), std::runtime_error, "Initial gradient norm is not finite.");

    // 3. main loop.
    dx_ = make_buffer<Scalar, Device>(problem.fused_gradients().shape());

    auto dx = dx_.view();
    auto g = problem.fused_gradients().as_const();

    for (; iteration < criteria.max_iterations_; ++iteration) {
      callback(result);

      // Apply the preconditioner.
      auto [recompute_analyz, hessian] = update_hessian_();
      recompute_analyz |= iteration == 0;
      if (recompute_analyz) {
        solver.compute(hessian);
      } else {
        solver.factorize();
      }

      // Solve the linear system: H dx = g
      try {
        sparse::convergence_criteria<Scalar> solver_criteria{solver_max_iter_, inexact_};
        solver.solve(dx, g, solver_criteria);
      } catch (const std::exception& e) {
        double grad_norm = blas.norm(g);
        double dx_norm = blas.norm(dx);
        fprintf(stderr, "Trace: |g| = %g, |dx| = %g\n", grad_norm, dx_norm);
        throw;
      }

      // Launch linesearcher.
      auto [ls_result, ls_step_size] = ls.search(problem, dx, 1);
      MATHPRIM_UNUSED(ls_result);
      MATHPRIM_UNUSED(ls_step_size);

      Scalar new_value = problem.current_value();
      last_change = value - new_value;
      value = new_value;
      grad_norm = blas.norm(grads);
      converged = grad_norm < criteria.tol_grad_;
      converged |= (last_change < criteria.tol_change_ && last_change >= 0);

      MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(value), std::runtime_error,
                                    "At step " << iteration << ", value is not finite.");
      MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(grad_norm), std::runtime_error,
                                    "At step " << iteration << ", gradient norm is not finite.");

      if (converged) {
        break;
      }
    }

    return result;
  }

  void set_inexact(Scalar inexact) { inexact_ = inexact; }
  void set_solver_max_iter(index_t max_iter) { solver_max_iter_ = max_iter; }
  SparseSolver& solver() { return solver_; }
  const SparseSolver& solver() const { return solver_; }
  Blas& blas() { return blas_; }
  const Blas& blas() const { return blas_; }
  Linesearcher& linesearcher() { return linesearcher_; }
  const Linesearcher& linesearcher() const { return linesearcher_; }

protected:
  SparseSolver solver_;

  // Inexact CG.
  Scalar inexact_ = 1e-2;
  index_t solver_max_iter_ = 1024;

  Linesearcher linesearcher_;
  Blas blas_;
  sparse_view hessian_;
  hessian_fn update_hessian_;
  contiguous_vector_buffer<Scalar, Device> dx_{};
};
}  // namespace mathprim::optim