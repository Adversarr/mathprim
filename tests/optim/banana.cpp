#include "mathprim/optim/ex_probs/banana.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include <iostream>

using namespace mathprim;

int main() {
  optim::ex_probs::banana_problem<double, device::cpu> problem(10, 10);
  problem.setup();
  std::cout << problem.eval_value() << std::endl;

  optim::gradient_descent_optimizer<double, device::cpu, blas::cpu_eigen<double>> gd;
  gd.learning_rate_ = 0.001;
  gd.momentum_ = 0.1;
  gd.nesterov_ = true;
  gd.stopping_criteria_.max_iterations_ = 8192;
  gd.stopping_criteria_.tol_change_ = 0;
  gd.stopping_criteria_.tol_grad_ = 1e-7;

  std::cout << gd.optimize(problem, [&problem](auto& result) {
    if (result.iterations_ % 256 == 0){
      std::cout << result << std::endl;
      std::cout << "x=" << eigen_support::cmap(problem.at(0).value()).transpose() << std::endl;
    }
  }) << std::endl;
  std::cout << "-------------------" << std::endl;

  optim::adamw_optimizer<double, device::cpu, blas::cpu_eigen<double>> adamw;
  adamw.learning_rate_ = 1e-3;
  adamw.beta1_ = 0.9;
  adamw.beta2_ = 0.99;
  adamw.stopping_criteria_.max_iterations_ = 8192;
  adamw.stopping_criteria_.tol_change_ = 0;
  adamw.stopping_criteria_.tol_grad_ = 1e-7;
  problem.setup();
  std::cout << adamw.optimize(problem, [&problem](auto& result) {
    if (result.iterations_ % 256 == 0) {
      std::cout << result << std::endl;
      std::cout << "x=" << eigen_support::cmap(problem.at(0).value()).transpose() << std::endl;
    }
  }) << std::endl;

  return 0;
}
