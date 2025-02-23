#include <benchmark/benchmark.h>

#include <Eigen/Sparse>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>

#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/sparse/blas/cholmod.hpp"
#include "mathprim/sparse/blas/cusparse.hpp"

using namespace mathprim;

template <typename blas_impl>
static void work(benchmark::State &state) {
  int dsize = state.range(0);
  // 1 D laplacian
  const int rows = dsize, cols = dsize, nnz = dsize * 3 - 2;

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  for (int i = 0; i < rows; i++) {
    if (i == 0) {
      row_ptr[i] = 0;
      col_idx[i * 3] = 0;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f + i;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      values.as_const(), row_ptr.as_const(), col_idx.as_const(), rows, cols, nnz, sparse::sparse_property::general);

  using linear_op
      = iterative_solver::sparse_matrix<sparse::blas::naive<float, sparse::sparse_format::csr, par::openmp>>;
  using preconditioner
      = iterative_solver::diagonal_preconditioner<float, device::cpu, sparse::sparse_format::csr, blas_impl>;
  iterative_solver::cg<float, device::cpu, linear_op, blas_impl, preconditioner> cg{linear_op{mat}, blas_impl{},
                                                                                    preconditioner{mat}};

  auto b = make_buffer<float>(rows);
  auto x = make_buffer<float>(rows);
  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());

    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });

    state.ResumeTiming();
    auto result = cg.apply(b.view(), x.view(),
                           {
                             .max_iterations_ = dsize * 4,
                             .norm_tol_ = 1e-6f,
                           });
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}

static void work_chol(benchmark::State &state) {
  int dsize = state.range(0);
  // 1 D laplacian
  const int rows = dsize, cols = dsize, nnz = dsize * 3 - 2;

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  for (int i = 0; i < rows; i++) {
    if (i == 0) {
      row_ptr[i] = 0;
      col_idx[i * 3] = 0;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f + i;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      values.as_const(), row_ptr.as_const(), col_idx.as_const(), rows, cols, nnz, sparse::sparse_property::general);

  using linear_op = iterative_solver::sparse_matrix<sparse::blas::cholmod<float, sparse::sparse_format::csr>>;
  using preconditioner = iterative_solver::diagonal_preconditioner<float, device::cpu, sparse::sparse_format::csr,
                                                                   blas::cpu_blas<float>>;
  iterative_solver::cg<float, device::cpu, linear_op, blas::cpu_blas<float>, preconditioner> cg{
    linear_op{mat}, blas::cpu_blas<float>{}, preconditioner{mat}};

  auto b = make_buffer<float>(rows);
  auto x = make_buffer<float>(rows);
  // GT = ones.
  for (auto _ : state) {
    state.PauseTiming();
    par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
      xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());

    par::seq().run(make_shape(rows), [xv = x.view(), bv = b.view()](index_t i) {
      xv[i] = (i % 100 - 50) / 100.0f;
    });

    state.ResumeTiming();
    auto result = cg.apply(b.view(), x.view(),
                           {
                             .max_iterations_ = dsize * 4,
                             .norm_tol_ = 1e-6f,
                           });
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}

void work_cuda(benchmark::State &state) {
  int dsize = state.range(0);
  // 1 D laplacian
  const int rows = dsize, cols = dsize, nnz = dsize * 3 - 2;

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  for (int i = 0; i < rows; i++) {
    if (i == 0) {
      row_ptr[i] = 0;
      col_idx[i * 3] = 0;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f + i;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  auto d_csr_values = make_cuda_buffer<float>(nnz);
  auto d_csr_col_idx = make_cuda_buffer<index_t>(nnz);
  auto d_csr_row_ptr = make_cuda_buffer<index_t>(rows + 1);

  copy(d_csr_values.view(), values);
  copy(d_csr_col_idx.view(), col_idx);
  copy(d_csr_row_ptr.view(), row_ptr);

  sparse::basic_sparse_view<const float, device::cuda, sparse::sparse_format::csr> mat(
      d_csr_values.const_view(), d_csr_row_ptr.const_view(), d_csr_col_idx.const_view(), rows, cols, nnz,
      sparse::sparse_property::symmetric);

  using linear_op = iterative_solver::sparse_matrix<sparse::blas::cusparse<float, sparse::sparse_format::csr>>;
  using blas_t = blas::cublas<float>;
  using preconditioner
      = iterative_solver::diagonal_preconditioner<float, device::cuda, sparse::sparse_format::csr, blas_t>;
  iterative_solver::cg<float, device::cuda, linear_op, blas::cublas<float>, preconditioner> cg{linear_op{mat}, blas_t{},
                                                                                               preconditioner{mat}};

  auto d_b = make_cuda_buffer<float>(rows);
  auto d_x = make_cuda_buffer<float>(rows);
  auto parfor = par::cuda();
  for (auto _ : state) {
    state.PauseTiming();
    parfor.run(make_shape(rows), [d_xv = d_x.view()] __device__(index_t i) {
      d_xv[i] = 1.0f;
    });
    // b = A * x
    cg.linear_operator().apply(1.0f, d_x.view(), 0.0f, d_b.view());

    parfor.run(make_shape(rows), [d_xv = d_x.view(), d_bv = d_b.view()] __device__(index_t i) {
      d_xv[i] = (i % 100 - 50) / 100.0f;
    });
    parfor.sync();
    state.ResumeTiming();
    auto result = cg.apply(d_b.view(), d_x.view(),
                           {
                             .max_iterations_ = dsize * 4,
                             .norm_tol_ = 1e-6f,
                           });
    parfor.sync();
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
}

void work_eigen_naive(benchmark::State &state) {
  int dsize = state.range(0);
  // 1 D laplacian
  const int rows = dsize, cols = dsize, nnz = dsize * 3 - 2;

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  for (int i = 0; i < rows; i++) {
    if (i == 0) {
      row_ptr[i] = 0;
      col_idx[i * 3] = 0;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f + i;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f + i;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      values.as_const(), row_ptr.as_const(), col_idx.as_const(), rows, cols, nnz, sparse::sparse_property::general);

  auto ei_mat = eigen_support::map(mat);

  Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor, index_t>, Eigen::Upper | Eigen::Lower,
                           Eigen::DiagonalPreconditioner<float>>
      cg;
  cg.compute(ei_mat);

  auto b = make_buffer<float>(rows);
  auto x = make_buffer<float>(rows);
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

    cg.setTolerance(1e-6 / bv.norm());
    state.ResumeTiming();
    xv = cg.solveWithGuess(bv, xv);
    state.SetLabel(std::to_string(cg.iterations()));
  }
}

BENCHMARK(work_eigen_naive)->Range(1 << 10, 1 << 16);
BENCHMARK(work_cuda)->Range(1 << 10, 1 << 16);
BENCHMARK_TEMPLATE(work, blas::cpu_handmade<float>)->Range(1 << 10, 1 << 16);
BENCHMARK_TEMPLATE(work, blas::cpu_blas<float>)->Range(1 << 10, 1 << 16);
BENCHMARK_TEMPLATE(work, blas::cpu_eigen<float>)->Range(1 << 10, 1 << 16);
BENCHMARK(work_chol)->Range(1 << 10, 1 << 16);

BENCHMARK_MAIN();