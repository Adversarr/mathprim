#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/optim/ex_probs/quad.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include <cstdlib>
#include <iostream>

using namespace mathprim;
using namespace mathprim::optim;


int main() {
  ex_probs::quad_problem_cu<double> problem(10, 2);
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
