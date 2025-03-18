#pragma once
#include <Eigen/IterativeLinearSolvers>

#include "mathprim/blas/cpu_handmade.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, sparse::sparse_format Format>
class wrap_eigen_sparse_map : public basic_linear_operator<wrap_eigen_sparse_map<Scalar, Format>, Scalar, device::cpu> {
  using map_type = Eigen::Map<const eigen_support::internal::eigen_sparse_format_t<Scalar, Format>>;
  using base = basic_linear_operator<wrap_eigen_sparse_map<Scalar, Format>, Scalar, device::cpu>;
  friend base;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = basic_sparse_view<const Scalar, device::cpu, Format>;

public:
  wrap_eigen_sparse_map(const_sparse mat) :  // NOLINT(google-explicit-constructor)
      base(mat.rows(), mat.cols()), mat_(eigen_support::map(mat)) {}

  wrap_eigen_sparse_map(wrap_eigen_sparse_map&&) = default;

  void apply_impl(Scalar alpha, const_vector x, Scalar beta, vector_type y) {
    auto x_map = eigen_support::cmap(x);
    auto y_map = eigen_support::cmap(y);
    if (beta == 0) {
      y_map.noalias() = mat_ * x_map;
    } else {
      y_map *= beta;
      y_map.noalias() += alpha * mat_ * x_map;
    }
  }

  void apply_transpose_impl(Scalar alpha, const_vector x, Scalar beta, vector_type y) {
    auto x_map = eigen_support::cmap(x);
    auto y_map = eigen_support::cmap(y);
    if (beta == 0) {
      y_map.noalias() = mat_.transpose() * x_map;
    } else {
      y_map *= beta;
      y_map.noalias() += alpha * mat_.transpose() * x_map;
    }
  }

  map_type matrix_map() const noexcept {
    return mat_;
  }

private:
  map_type mat_;  // Eigen sparse matrix map
};

template <typename EigenSolver, typename Scalar, sparse::sparse_format Format>
class basic_eigen_iterative_solver
    : public basic_iterative_solver<basic_eigen_iterative_solver<EigenSolver, Scalar, Format>, Scalar, device::cpu,
                                    wrap_eigen_sparse_map<Scalar, Format>> {
public:
  using base = basic_iterative_solver<basic_eigen_iterative_solver<EigenSolver, Scalar, Format>, Scalar, device::cpu,
                                      wrap_eigen_sparse_map<Scalar, Format>>;
  friend base;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using linear_operator_type = typename base::linear_operator_type;
  using results_type = typename base::results_type;
  using parameters_type = typename base::parameters_type;
  explicit basic_eigen_iterative_solver(linear_operator_type matrix) :
      base(std::move(matrix)) {
    compute();
  }

  void compute() {
    solver_ = std::make_unique<EigenSolver>();
    solver_->compute(base::linear_operator().matrix_map());
  }

private:
  template <typename Callback>
  MATHPRIM_NOINLINE results_type solve_impl(vector_type x, const_vector b, const parameters_type& params = {},
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