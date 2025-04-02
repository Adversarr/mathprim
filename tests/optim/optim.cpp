#include <benchmark/benchmark.h>

#include <mathprim/blas/cpu_blas.hpp>

#include "LBFGS.h"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/optim/ex_probs/banana.hpp"
#include "mathprim/optim/optimizer/adamw.hpp"
#include "mathprim/optim/optimizer/gradient_descent.hpp"
#include "mathprim/optim/optimizer/l_bfgs.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace LBFGSpp;

class Rosenbrock {
private:
  int n;
  ptrdiff_t ncalls;

public:
  Rosenbrock(int n_) : n(n_), ncalls(0) {}
  double operator()(const VectorXd& x, VectorXd& grad) {
    //        std::cout << x << std::endl;
    ncalls += 1;
    // for (index_t i = 0; i < dsize - 1; i++) {
    //   Scalar nonlinear = (x(i) * x(i) - x(i + 1));
    //   base::accumulate_loss(nonlinear * nonlinear * difficulty_);
    //   Scalar linear = 1 - x(i);
    //   base::accumulate_loss(linear * linear);
    //   grad(i) += 4 * difficulty_ * nonlinear * x(i) - 2 * linear;
    //   grad(i + 1) += -2 * difficulty_ * nonlinear;
    // }
    double fx = 0.0;
    grad.setZero();
    for (int i = 0; i < n - 1; ++i) {
      double nonlinear = x(i) * x(i) - x(i+1);
      double linear = 1 - x(i);
      fx += 100 * nonlinear * nonlinear + linear * linear;
      grad(i) += 400 * nonlinear * x(i) - 2 * linear;
      grad(i+1) += -200 * nonlinear;
    }
    assert(!std::isnan(fx));
    return fx;
  }

  ptrdiff_t get_ncalls() { return ncalls; }
};

template <template <class> typename LineSearch>
void lbfgspp(benchmark::State& state) {
  int n = state.range(0);
  LBFGSParam<double> param;
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
  param.max_linesearch = 256;
  param.epsilon = 1e-5;
  param.epsilon_rel = 0;
  param.m = 5;


  for (auto _ : state) {
    Rosenbrock fun_backtrack(n);
    LBFGSSolver<double, LineSearch> solver(param);
    VectorXd x, x0 = VectorXd::Zero(n);
    double fx;

    x = x0;
    int niter = solver.minimize(fun_backtrack, x, fx);

    state.SetLabel(std::to_string(niter) + ":" + std::to_string(fun_backtrack.get_ncalls()));
  }
}

void ours(benchmark::State& state) {
  int n = state.range(0);

  using namespace mathprim;
  optim::ex_probs::banana_problem<double, device::cpu> problem(n, 100);
  problem.setup();

  optim::l_bfgs_optimizer<double, device::cpu, blas::cpu_eigen<double>> l_bfgs;
  for (auto _ : state) {
    problem.setup();
    l_bfgs.criteria().max_iterations_ = 5000;
    l_bfgs.criteria().tol_grad_ = 1e-5;
    l_bfgs.memory_size_ = 6;
    auto info = l_bfgs.optimize(problem);
    state.SetLabel(std::to_string(info.iterations_) + ":" + std::to_string(problem.eval_cnt()));
  }
}

std::vector<int64_t> sizes = {128, 256};

BENCHMARK(ours)->ArgsProduct({sizes})->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(lbfgspp, LineSearchBacktracking)->ArgsProduct({sizes})->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
