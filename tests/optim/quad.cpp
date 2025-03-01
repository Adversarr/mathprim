#include "mathprim/optim/ex_probs/quad.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/optim/linesearcher/backtracking.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include "mathprim/optim/optimizer/l_bfgs.hpp"
#include <iostream>

using namespace mathprim;

int main() {
  optim::ex_probs::quad_problem<double> problem(10, 2);
  problem.setup();
  std::cout << problem.eval_value() << std::endl;

  {
    optim::gradient_descent_optimizer<double, device::cpu, blas::cpu_eigen<double>> gd;
    gd.learning_rate_ = 0.05;
    gd.momentum_ = 0.5;
    gd.nesterov_ = true;

    gd.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    });
  }

  {
    optim::adamw_optimizer<double, device::cpu, blas::cpu_eigen<double>> adamw;
    adamw.learning_rate_ = 1e-1;
    adamw.beta1_ = 0.9;
    adamw.beta2_ = 0.95;
    adamw.criteria().max_iterations_ = 1000;
    problem.setup();
    adamw.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    });
  }

  {
    using bt = optim::backtracking_linesearcher<double, device::cpu, blas::cpu_eigen<double>>;
    optim::gradient_descent_optimizer<double, device::cpu, blas::cpu_eigen<double>, bt> gd;
    problem.setup();
    gd.linesearcher().stopping_criteria_.max_iterations_ = 10;
    gd.learning_rate_ = 0.5;

    gd.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    });
  }

  {
    optim::l_bfgs_optimizer<double, device::cpu, blas::cpu_eigen<double>> l_bfgs;
    problem.setup();
    l_bfgs.criteria().max_iterations_ = 1000;
    l_bfgs.memory_size_ = 10;
    l_bfgs.stopping_criteria_.max_iterations_ = 100;
    l_bfgs.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    });
  }


  {
    using ls = optim::backtracking_linesearcher<double, device::cpu, blas::cpu_eigen<double>>;
    using pr = optim::l_bfgs_preconditioner_identity<double, device::cpu, blas::cpu_eigen<double>>;
    optim::l_bfgs_optimizer<double, device::cpu, blas::cpu_eigen<double>, ls, pr> l_bfgs;
    problem.setup();
    l_bfgs.criteria().max_iterations_ = 1000;
    l_bfgs.memory_size_ = 5;
    l_bfgs.stopping_criteria_.max_iterations_ = 100;
    l_bfgs.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    });
  }

  return 0;
}
