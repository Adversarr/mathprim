#include "mathprim/optim/ex_probs/quad.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/optim/linesearcher/backtracking.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include "mathprim/optim/optimizer/l_bfgs.hpp"
#include "mathprim/optim/optimizer/newton.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include <iostream>

using namespace mathprim;

int main() {
  optim::ex_probs::quad_problem<double> problem(10, 1);
  problem.setup();
  std::cout << problem.eval_value() << std::endl;

  {
    optim::gradient_descent_optimizer<double, device::cpu, blas::cpu_eigen<double>> gd;
    gd.learning_rate_ = 0.05;
    gd.momentum_ = 0.5;
    gd.nesterov_ = true;

    std::cout << gd.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    }) << std::endl;
  }

  {
    optim::adamw_optimizer<double, device::cpu, blas::cpu_eigen<double>> adamw;
    adamw.learning_rate_ = 1e-1;
    adamw.beta1_ = 0.9;
    adamw.beta2_ = 0.95;
    adamw.criteria().max_iterations_ = 1000;
    problem.setup();
    std::cout << adamw.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    }) << std::endl;
  }

  {
    using bt = optim::backtracking_linesearcher<double, device::cpu, blas::cpu_eigen<double>>;
    optim::gradient_descent_optimizer<double, device::cpu, blas::cpu_eigen<double>, bt> gd;
    problem.setup();
    gd.linesearcher().stopping_criteria_.max_iterations_ = 100;
    gd.learning_rate_ = 0.5;

    std::cout << gd.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    }) << std::endl;
  }

  {
    optim::l_bfgs_optimizer<double, device::cpu, blas::cpu_eigen<double>> l_bfgs;
    problem.setup();
    l_bfgs.criteria().max_iterations_ = 1000;
    l_bfgs.memory_size_ = 10;
    l_bfgs.stopping_criteria_.max_iterations_ = 100;
    std::cout << l_bfgs.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    }) << std::endl;
  }


  {
    using ls = optim::backtracking_linesearcher<double, device::cpu, blas::cpu_eigen<double>>;
    using pr = optim::l_bfgs_preconditioner_identity<double, device::cpu, blas::cpu_eigen<double>>;
    optim::l_bfgs_optimizer<double, device::cpu, blas::cpu_eigen<double>, ls, pr> l_bfgs;
    problem.setup();
    l_bfgs.criteria().max_iterations_ = 1000;
    l_bfgs.memory_size_ = 5;
    l_bfgs.stopping_criteria_.max_iterations_ = 100;
    std::cout << l_bfgs.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    }) << std::endl;
  }

  {
    Eigen::SparseMatrix<double, Eigen::RowMajor> hessian(10, 10);
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        hessian.insert(i, j) = problem.A_[0].view()(i, j);
      }
    }
    hessian.makeCompressed();
    using ls = optim::backtracking_linesearcher<double, device::cpu, blas::cpu_eigen<double>>;
    using solver = sparse::iterative::cg<double, device::cpu,
                                         mathprim::sparse::blas::eigen<double, mathprim::sparse::sparse_format::csr>,
                                         blas::cpu_eigen<double>>;
    optim::newton_optimizer<double, device::cpu, blas::cpu_eigen<double>, ls, solver> newton;
    newton.set_hessian_fn([&hessian]() {
      return std::make_pair(false, eigen_support::view(hessian).as_const());
    });
    problem.setup();

    std::cout << newton.optimize(problem, [](auto& result) {
      std::cout << result << std::endl;
    }) << std::endl;
  }

  return 0;
}
