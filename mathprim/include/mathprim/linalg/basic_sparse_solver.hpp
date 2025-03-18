#pragma once
#include "mathprim/sparse/basic_sparse.hpp"
namespace mathprim::sparse {

template <typename Scalar>
struct convergence_criteria {
  index_t max_iterations_;
  Scalar norm_tol_;
  convergence_criteria(index_t max_iterations = 1 << 10,  // NOLINT(google-explicit-constructor)
                       Scalar norm_tol = 1e-3) :
      max_iterations_(max_iterations), norm_tol_(norm_tol) {}
  MATHPRIM_INTERNAL_COPY(convergence_criteria, default);
};

template <typename Scalar>
struct convergence_result {
  index_t iterations_ = {1 << 10};                         ///< number of iterations.
  Scalar norm_ = {std::numeric_limits<float>::epsilon()};  ///< l2 norm of the residual. (norm(r))

  convergence_result(index_t iterations = 1 << 10,  // NOLINT(google-explicit-constructor)
                     Scalar norm = 1e-3) :
      iterations_(iterations), norm_(norm) {}
  MATHPRIM_INTERNAL_COPY(convergence_result, default);
};

template <typename Derived, typename Scalar, typename Device, sparse::sparse_format Compression>
class basic_sparse_solver {
public:
  using vector_view = contiguous_view<Scalar, dshape<1>, Device>;
  using const_vector = contiguous_view<const Scalar, dshape<1>, Device>;
  using matrix_view = contiguous_view<Scalar, dshape<2>, Device>;
  using const_matrix_view = contiguous_view<const Scalar, dshape<2>, Device>;

  using sparse_view = basic_sparse_view<Scalar, Device, Compression>;
  using const_sparse = basic_sparse_view<const Scalar, Device, Compression>;

  using results_type = convergence_result<Scalar>;
  using parameters_type = convergence_criteria<Scalar>;

  basic_sparse_solver() = default;
  explicit basic_sparse_solver(const const_sparse& mat) : mat_(mat) {}
  MATHPRIM_INTERNAL_MOVE(basic_sparse_solver, default);
  MATHPRIM_INTERNAL_COPY(basic_sparse_solver, delete);

  struct no_op {
    inline void operator()(index_t /* iter */, Scalar /* norm */) const noexcept {}
  };

  // Solve the linear system.
  template <typename Callback = no_op>
  results_type solve(vector_view x, const_vector b, const parameters_type& params = {}, Callback&& cb = {}) {
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.cols() == x.size(), std::runtime_error, "Invalid size of x");
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.rows() == b.size(), std::runtime_error, "Invalid size of b");
    return derived().solve_impl(x, b, params, std::forward<Callback>(cb));
  }

  // Solve the linear system. (Matrix version)
  void vsolve(matrix_view lhs, const_matrix_view rhs, const parameters_type& params = {}) {
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.cols() == lhs.shape(0), std::runtime_error, "Invalid size of lhs");
    MATHPRIM_INTERNAL_CHECK_THROW(mat_.rows() == rhs.shape(0), std::runtime_error, "Invalid size of rhs");
    MATHPRIM_INTERNAL_CHECK_THROW(lhs.shape(1) == rhs.shape(1), std::runtime_error, "Invalid size of rhs");
    derived().vsolve_impl(lhs, rhs, params);
  }

  void vsolve_impl(matrix_view /* lhs */, const_matrix_view /* rhs */, const parameters_type& /* params */) {
    throw std::runtime_error("This solver does not support vectorized solve.");
  }

  // Analyze the non-zero pattern of matrix.
  Derived& analyze(const const_sparse& mat) {
    mat_ = mat;
    derived().analyze_impl();
    return derived();
  }

  Derived& factorize() {
    derived().factorize_impl();
    return derived();
  }

  Derived& compute(const const_sparse& mat) { return analyze(mat).factorize(); }

  Derived& derived() { return *static_cast<Derived*>(this); }
  const Derived& derived() const { return *static_cast<const Derived*>(this); }
  const_sparse mat_;
};
}  // namespace mathprim::sparse