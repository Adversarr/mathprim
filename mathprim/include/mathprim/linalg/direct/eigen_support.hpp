#pragma once
#include "mathprim/linalg/direct/direct.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

namespace mathprim::sparse::direct {

template <typename EigenSolver, typename Scalar, sparse::sparse_format Format>
class basic_eigen_direct_solver
    : public basic_direct_solver<basic_eigen_direct_solver<EigenSolver, Scalar, Format>, Scalar, Format, device::cpu> {
public:
  using base = basic_direct_solver<basic_eigen_direct_solver<EigenSolver, Scalar, Format>, Scalar, Format, device::cpu>;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  friend base;

  basic_eigen_direct_solver() = default;
  explicit basic_eigen_direct_solver(const_matrix_view mat) : base(mat) {
    base::compute(mat);
  }

  basic_eigen_direct_solver(basic_eigen_direct_solver&& other) : base(other.mat_) {
    std::swap(solver_, other.solver_);
  }

  ~basic_eigen_direct_solver() {
    reset();
  }

private:
  void reset() {
    solver_.reset();
  }

  void analyze_impl() {
    solver_ = std::make_unique<EigenSolver>();
    solver_->analyzePattern(eigen_support::map(mat_));
  }

  void factorize_impl() {
    MATHPRIM_INTERNAL_CHECK_THROW(solver_, std::runtime_error, "The solver is not initialized.");
    solver_->factorize(eigen_support::map(mat_));
  }

  void solve_impl(vector_view lhs, const_vector rhs) {
    MATHPRIM_INTERNAL_CHECK_THROW(solver_, std::runtime_error, "The solver is not initialized.");
    eigen_support::cmap(lhs) = solver_->solve(eigen_support::cmap(rhs));
  }

protected:
  using base::mat_;
  std::unique_ptr<EigenSolver> solver_;
};

template <typename Scalar, sparse::sparse_format Format>
using eigen_simplicial_ldlt
    = basic_eigen_direct_solver<Eigen::SimplicialLDLT<eigen_support::internal::eigen_sparse_format_t<Scalar, Format>>,
                                Scalar, Format>;

template <typename Scalar, sparse::sparse_format Format>
using eigen_simplicial_llt
    = basic_eigen_direct_solver<Eigen::SimplicialLLT<eigen_support::internal::eigen_sparse_format_t<Scalar, Format>>,
                                Scalar, Format>;

template <typename Scalar, sparse::sparse_format Format>
using eigen_simplicial_chol = eigen_simplicial_ldlt<Scalar, Format>;
}  // namespace mathprim::sparse::direct