#include <benchmark/benchmark.h>

#include <Eigen/Sparse>
#include <iostream>
#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/blas/cpu_eigen.hpp>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/cholmod.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/sparse/systems/laplace.hpp>

#include "mathprim/linalg/iterative/precond/eigen_support.hpp"
#include "mathprim/linalg/iterative/solver/eigen_support.hpp"

using namespace mathprim;
using Scalar = double;
template <typename BlasImpl>
static void work(benchmark::State &state) {
  int dsize = state.range(0);

  sparse::convergence_criteria<Scalar> criteria{dsize * dsize, 1e-6f};
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();
  using SparseBlas = sparse::blas::naive<Scalar, sparse::sparse_format::csr, par::openmp>;
  using preconditioner
      = sparse::iterative::diagonal_preconditioner<Scalar, device::cpu, sparse::sparse_format::csr, BlasImpl>;
  sparse::iterative::cg<Scalar, device::cpu, SparseBlas, BlasImpl, preconditioner> cg{mat};

  auto b = make_buffer<Scalar>(rows); b.fill_bytes(0);
  auto x = make_buffer<Scalar>(rows);
  auto r = make_buffer<Scalar>(rows);
  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().gemv(1.0f, x.view(), 0.0f, b.view());

    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });
    state.ResumeTiming();
    auto result = cg.solve(x.view(), b.view(), criteria);
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
    cg.blas().copy(r.view(), b.view()); // r = b
    cg.linear_operator().gemv(-1, x.view(), 1, r.view()); // r = b - A * x
    auto norm = cg.blas().norm(r.view()) / cg.blas().norm(b.view());

    state.SetLabel(std::to_string(result.iterations_) + " " + std::to_string(norm));
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}
template <typename BlasImpl>
static void work_ic(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();


  using SparseBlas = sparse::blas::naive<Scalar, sparse::sparse_format::csr>;
  using preconditioner = sparse::iterative::eigen_ichol<Scalar, sparse::sparse_format::csr>;
  sparse::iterative::cg<Scalar, device::cpu, SparseBlas, BlasImpl, preconditioner> cg;
  cg.compute(mat);

  auto b = make_buffer<Scalar>(rows); b.fill_bytes(0);
  auto x = make_buffer<Scalar>(rows);
  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().gemv(1.0f, x.view(), 0.0f, b.view());
    sparse::convergence_criteria<Scalar> criteria{dsize * dsize, 1e-6f};

    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });
    state.ResumeTiming();
    auto result = cg.solve(x.view(), b.view(), criteria);
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}

template <typename BlasImpl>
static void work2(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();

  using SparseBlas = sparse::blas::naive<Scalar, sparse::sparse_format::csr>;
  using preconditioner = sparse::iterative::eigen_ichol<Scalar, sparse::sparse_format::csr>;

  sparse::iterative::cg<Scalar, device::cpu, SparseBlas, BlasImpl, preconditioner> cg{mat};

  auto b = make_buffer<Scalar>(rows); b.fill_bytes(0);
  auto x = make_buffer<Scalar>(rows);
  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().gemv(1.0f, x.view(), 0.0f, b.view());
    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });

    sparse::convergence_criteria<Scalar> criteria{dsize * dsize, 1e-6f};
    state.ResumeTiming();
    auto result = cg.solve(x.view(), b.view(), criteria);
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}

static void work_chol(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();

  using linear_op = sparse::blas::cholmod<Scalar, sparse::sparse_format::csr>;
  using preconditioner = sparse::iterative::diagonal_preconditioner<Scalar, device::cpu, sparse::sparse_format::csr,
                                                                    blas::cpu_blas<Scalar>>;
  sparse::iterative::cg<Scalar, device::cpu, linear_op, blas::cpu_blas<Scalar>, preconditioner> cg{mat};

  auto b = make_buffer<Scalar>(rows); b.fill_bytes(0);
  auto x = make_buffer<Scalar>(rows);
  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().gemv(1.0f, x.view(), 0.0f, b.view());

    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });

    state.ResumeTiming();
    sparse::convergence_criteria<Scalar> criteria{dsize * dsize, 1e-6f};

    auto result = cg.solve(x.view(), b.view(), criteria);
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}

void work_eigen_naive(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();
  auto ei_mat = eigen_support::map(mat);

  Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar, Eigen::RowMajor, index_t>, Eigen::Upper | Eigen::Lower,
                           Eigen::IncompleteCholesky<Scalar, Eigen::Upper | Eigen::Lower>>
      cg;
  cg.compute(ei_mat);

  auto b = make_buffer<Scalar>(rows); b.fill_bytes(0);
  auto x = make_buffer<Scalar>(rows);
  auto bv = eigen_support::map(b.view());
  auto xv = eigen_support::map(x.view());

  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    xv.setOnes();
    bv.noalias() = ei_mat * xv;
    for (index_t i = 0; i < rows; i++) {
      xv[i] = (i % 100 - 50) / 100.0f;
    }
    state.ResumeTiming();
    cg.setTolerance(1e-6);
    xv = cg.solveWithGuess(bv, xv);
    state.SetLabel(std::to_string(cg.iterations()));
  }
}

void work_eigen_wrapped(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();
  auto ei_mat = eigen_support::map(mat);

  using EigenSolver
      = Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar, Eigen::RowMajor, index_t>, Eigen::Upper | Eigen::Lower,
                                 Eigen::DiagonalPreconditioner<Scalar>>;

  sparse::iterative::basic_eigen_iterative_solver<EigenSolver, Scalar, sparse::sparse_format::csr> cg{mat};

  auto b = make_buffer<Scalar>(rows); b.fill_bytes(0);
  auto x = make_buffer<Scalar>(rows);

  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().gemv(1.0f, x.view(), 0.0f, b.view());

    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });

    state.ResumeTiming();
    sparse::convergence_criteria<Scalar> criteria{dsize * dsize, 1e-6f};

    auto result = cg.solve(x.view(), b.view(), criteria);
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
}

#ifdef NDEBUG
constexpr index_t lower = 1 << 4, upper = 1 << 8;
#else
constexpr index_t lower = 1 << 4, upper = 1 << 6;
#endif
BENCHMARK_TEMPLATE(work, blas::cpu_blas<Scalar>)->Range(lower, upper);
BENCHMARK_TEMPLATE(work, blas::cpu_eigen<Scalar>)->Range(lower, upper);
BENCHMARK_TEMPLATE(work, blas::cpu_handmade<Scalar>)->Range(lower, upper);
BENCHMARK(work_eigen_naive)->Range(lower, upper);
BENCHMARK(work_eigen_wrapped)->Range(lower, upper);
BENCHMARK(work_chol)->Range(lower, upper);
BENCHMARK_TEMPLATE(work_ic, blas::cpu_eigen<Scalar>)->Range(lower, upper);
BENCHMARK_TEMPLATE(work2, blas::cpu_blas<Scalar>)->Range(lower, upper);
BENCHMARK_TEMPLATE(work2, blas::cpu_handmade<Scalar>)->Range(lower, upper);
BENCHMARK_TEMPLATE(work2, blas::cpu_eigen<Scalar>)->Range(lower, upper);
BENCHMARK_MAIN();
