#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include <iostream>
#include <cstdlib>

using namespace mathprim;
using namespace mathprim::optim;

template <typename Scalar, typename T> void set_values(T &A, index_t dsize) {
  par::cuda().run(A.shape(), [A, dsize] __device__(auto ij) {
    auto [i, j] = ij;
    A(i, j) =  i == j ? dsize : static_cast<Scalar>(i + j) / dsize;
  });
}

// All optimizer should work on this simple problem
template <typename Scalar>
class quad_problem : public basic_problem<quad_problem<Scalar>, Scalar, device::cuda> {
public:
  using base = basic_problem<quad_problem<Scalar>, Scalar, device::cuda>;
  friend base;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using buffer_type = typename base::buffer_type;
  using parameter = typename base::parameter;

  // It formulates a quadratic problem with the following form:
  // f(x) = 0.5 * x^T * A * x
  explicit quad_problem(index_t dsize, index_t groups = 1) {
    // set A to randomly diagonal dominance
    for (index_t i = 0; i < groups; ++i) {
      A_.emplace_back(make_cuda_buffer<Scalar>(dsize, dsize));
      auto A = A_.back().view();
      set_values<Scalar>(A, dsize);
      parameters_.emplace_back(make_cuda_buffer<Scalar>(dsize));
      base::register_parameter(parameters_.back().view());
    }
  }

  quad_problem(quad_problem&&) noexcept = default;
  quad_problem& operator=(quad_problem&&) noexcept = default;

  void on_setup() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      // auto x = eigen_support::cmap(parameters_[i].view());
      // x.setRandom();
      par::cuda().run(parameters_[i].view().shape(), [v = parameters_[i].view()] __device__(auto i) {
        v(i) = static_cast<Scalar>(i) / v.size();
      });
    }
  }

  void eval_value_and_gradients_impl() {
    blas::cublas<Scalar> b;
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto A = A_[i].view();
      auto values = base::at(i).value();
      auto gradients = base::at(i).gradient();
      b.gemv(1, A, values, 0, gradients);
      base::accumulate_loss(0.5 * b.dot(values, gradients));
    }
  }

  void eval_value_impl() { eval_value_and_gradients_impl(); }

  void eval_gradients_impl() { eval_value_and_gradients_impl(); }

  std::vector<buffer_type> parameters_;
  std::vector<contiguous_matrix_buffer<Scalar, device::cuda>> A_;
};


int main() {
  quad_problem<double> problem(10, 2);
  problem.setup();
  std::cout << problem.eval_value() << std::endl;

  optim::gradient_descent_optimizer<double, device::cuda, blas::cublas<double>> gd;
  gd.learning_rate_ = 0.05;
  gd.momentum_ = 0.5;
  gd.nesterov_ = true;

  gd.optimize(problem, [](auto& result) {
    std::cout << result << std::endl;
  });

  optim::adamw_optimizer<double, device::cuda, blas::cublas<double>> adamw;
  adamw.learning_rate_ = 1e-1;
  adamw.beta1_ = 0.9;
  adamw.beta2_ = 0.95;
  adamw.criteria().max_iterations = 1000;
  problem.setup();
  adamw.optimize(problem, [](auto& result) {
    std::cout << result << std::endl;
  });

  return 0;
}
