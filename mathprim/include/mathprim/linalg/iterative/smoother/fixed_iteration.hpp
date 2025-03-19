#pragma once
#include "smoother.hpp"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, sparse_format Format, typename SparseBlas, typename Blas,
          typename Smoother>
class basic_fixed_iteration
    : public basic_iterative_solver<basic_fixed_iteration<Scalar, Device, Format, SparseBlas, Blas, Smoother>,
                                    Scalar, Device, SparseBlas> {
private:
  using this_type = basic_fixed_iteration<Scalar, Device, Format, SparseBlas, Blas, Smoother>;
  using base = basic_iterative_solver<this_type, Scalar, Device, SparseBlas>;
  using base2 = basic_sparse_solver<this_type, Scalar, Device, SparseBlas::compression>;
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

public:
  explicit basic_fixed_iteration(const_sparse matrix) : base(matrix) { this->compute({}); }

protected:
  void analyze_impl_impl() {
    smoother_.analyze(this->matrix());
    dx_ = make_buffer<Scalar, Device>(this->matrix().rows());
  }

  void factorize_impl_impl() { smoother_.factorize(); }

  template <typename Callback>
  results_type solve_impl(vector_view x, const_vector b, const parameters_type& params, Callback&& callback) {
    auto& blas = blas_;
    auto& smoother = smoother_;
    auto& matrix = this->linear_operator();
    auto& residual_buffer = base::residual_;
    if (!dx_ || dx_.size() != x.size()) {
      dx_ = make_buffer<Scalar, Device>(x.shape());
    }

    vector_view r = residual_buffer.view(), dx = dx_.view();
    const_vector cx = x.as_const(), cr = r.as_const();
    const Scalar b_norm = blas.norm(b);

    // initialize r <- b - A * x
    blas.copy(r, b);
    matrix.gemv(-1, cx, 1, r);

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
      matrix.gemv(-1, dx, 1, r);  // r <- r - A * (delta x)

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