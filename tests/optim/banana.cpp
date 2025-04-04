#include "mathprim/optim/ex_probs/banana.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include "mathprim/optim/optimizer/l_bfgs.hpp"
#include "mathprim/optim/optimizer/ncg.hpp"
#include <iostream>

using namespace mathprim;

int main() {
  optim::ex_probs::banana_problem<double, device::cpu> problem(10, 40);
  problem.setup();
  std::cout << problem.eval_value() << std::endl;

  optim::gradient_descent_optimizer<double, device::cpu, blas::cpu_eigen<double>> gd;
  gd.learning_rate_ = 0.001;
  gd.momentum_ = 0.1;
  gd.nesterov_ = true;
  gd.stopping_criteria_.max_iterations_ = 8192;
  gd.stopping_criteria_.tol_change_ = 0;
  gd.stopping_criteria_.tol_grad_ = 1e-7;

  std::cout << gd.optimize(problem) << std::endl;
  std::cout << "-------------------" << std::endl;
  {
    optim::adamw_optimizer<double, device::cpu, blas::cpu_eigen<double>> adamw;
    adamw.learning_rate_ = 1e-3;
    adamw.beta1_ = 0.9;
    adamw.beta2_ = 0.99;
    adamw.stopping_criteria_.max_iterations_ = 8192;
    adamw.stopping_criteria_.tol_change_ = 0;
    adamw.stopping_criteria_.tol_grad_ = 1e-7;
    problem.setup();
    std::cout << adamw.optimize(problem) << std::endl;
  }

  {
    optim::l_bfgs_optimizer<double, device::cpu, blas::cpu_eigen<double>> l_bfgs;
    problem.setup();
    l_bfgs.criteria().max_iterations_ = 1000;
    l_bfgs.memory_size_ = 5;
    l_bfgs.stopping_criteria_.max_iterations_ = 1000;
    std::cout << l_bfgs.optimize(problem) << std::endl;
  }

  for (int i = 0; i < 6; ++ i) {
    std::cout << "Nonlinear Conjugate Gradient " << i << std::endl;
    using ls = optim::backtracking_linesearcher<double, device::cpu, blas::cpu_eigen<double>>;
    using opt = optim::ncg_optimizer<double, device::cpu, blas::cpu_eigen<double>, ls>;
    opt ncg;
    ncg.stopping_criteria_.max_iterations_ = 1000;
    ncg.stopping_criteria_.tol_grad_ = 1e-6;
    ncg.strategy_ = static_cast<optim::ncg_strategy>(i);
    problem.setup();
    std::cout << ncg.optimize(problem) << std::endl;
    std::cout << "x=" << eigen_support::cmap(problem.at(0).value()).transpose() << std::endl;
    std::cout << "-------------------" << std::endl;
  }

  return 0;
}
