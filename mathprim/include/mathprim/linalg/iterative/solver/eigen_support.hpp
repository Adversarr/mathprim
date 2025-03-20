#pragma once
#include <Eigen/IterativeLinearSolvers>

#include "mathprim/blas/cpu_handmade.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim::sparse::iterative {

template <typename EigenSolver, typename Scalar, sparse_format Format>
class basic_eigen_iterative_solver final
    : public basic_iterative_solver<basic_eigen_iterative_solver<EigenSolver, Scalar, Format>, Scalar, device::cpu,
                                    sparse::blas::eigen<Scalar, Format>> {
public:
  using this_type = basic_eigen_iterative_solver<EigenSolver, Scalar, Format>;
  using base = basic_iterative_solver<this_type, Scalar, device::cpu, sparse::blas::eigen<Scalar, Format>>;
  using base2 = basic_sparse_solver<this_type, Scalar, device::cpu, Format>;
  friend base;
  friend base2;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using sparse_view = typename base::sparse_view;
  using const_sparse = typename base::const_sparse;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  using results_type = convergence_result<Scalar>;
  using parameters_type = convergence_criteria<Scalar>;

  basic_eigen_iterative_solver() = default;
  explicit basic_eigen_iterative_solver(const_sparse matrix) : base(matrix) { this->compute(); }

protected:
  void analyze_impl_impl() {
    solver_ = std::make_unique<EigenSolver>();
    solver_->analyzePattern(eigen_support::map(this->matrix()));
  }

  void factorize_impl_impl() { solver_->factorize(eigen_support::map(this->matrix())); }

  template <typename Callback>
  results_type solve_impl(vector_view x, const_vector b, const parameters_type& params = {},
                                            Callback&& /* cb */ = {}) {
    MATHPRIM_INTERNAL_CHECK_THROW(solver_, std::runtime_error, "The solver is not initialized.");
    results_type res;
    auto b_map = eigen_support::cmap(b);
    auto x_map = eigen_support::cmap(x);

    solver_->setMaxIterations(params.max_iterations_);
    solver_->setTolerance(params.norm_tol_);
    x_map = solver_->solveWithGuess(b_map, x_map).eval();
    res.iterations_ = solver_->iterations();
    res.norm_ = solver_->error();
    return res;
  }

  std::unique_ptr<EigenSolver> solver_;
};
}  // namespace mathprim::sparse::iterative