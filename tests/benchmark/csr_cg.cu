#include <benchmark/benchmark.h>

#include <Eigen/Sparse>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/ilu_cusparse.hpp>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>

#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/linalg/iterative/precond/eigen_support.hpp"
#include "mathprim/linalg/iterative/precond/ic_cusparse.hpp"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/sparse/blas/cusparse.hpp"
#include "mathprim/sparse/systems/laplace.hpp"

using namespace mathprim;

template <typename BlasImpl>
static void work(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();

  using linear_op
      = sparse::iterative::sparse_matrix<sparse::blas::naive<float, sparse::sparse_format::csr, par::openmp>>;
  using preconditioner
      = sparse::iterative::diagonal_preconditioner<float, device::cpu, sparse::sparse_format::csr, BlasImpl>;
  sparse::iterative::cg<float, device::cpu, linear_op, BlasImpl, preconditioner> cg{linear_op{mat}, BlasImpl{},
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
                             .max_iterations_ = dsize * dsize,
                             .norm_tol_ = 1e-6f,
                           });
    state.SetLabel(std::to_string(result.iterations_));
    if (result.norm_ > 1e-6f) {
      state.SkipWithError("CG did not converge");
    }
  }
  // par::seq().run(make_shape(rows), [xv = x.view()](index_t i) { std::cout << xv[i] << std::endl; });
}


template <typename BlasImpl>
static void work_ic(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();

  using linear_op
      = sparse::iterative::sparse_matrix<sparse::blas::naive<float, sparse::sparse_format::csr, par::openmp>>;
  using preconditioner = sparse::iterative::eigen_ichol<float>;
  sparse::iterative::cg<float, device::cpu, linear_op, BlasImpl, preconditioner> cg{linear_op{mat}, BlasImpl{},
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
                             .max_iterations_ = dsize * dsize,
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
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto h_mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto d_mat_buf = h_mat_buf.to<device::cuda>();
  auto mat = d_mat_buf.const_view();
  auto rows = mat.rows();
  auto nnz = mat.nnz();

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  auto d_csr_values = make_cuda_buffer<float>(nnz);
  auto d_csr_col_idx = make_cuda_buffer<index_t>(nnz);
  auto d_csr_row_ptr = make_cuda_buffer<index_t>(rows + 1);

  copy(d_csr_values.view(), values);
  copy(d_csr_col_idx.view(), col_idx);
  copy(d_csr_row_ptr.view(), row_ptr);

  using linear_op = sparse::iterative::sparse_matrix<sparse::blas::cusparse<float, sparse::sparse_format::csr>>;
  using blas_t = blas::cublas<float>;
  using preconditioner
      = sparse::iterative::diagonal_preconditioner<float, device::cuda, sparse::sparse_format::csr, blas_t>;
  sparse::iterative::cg<float, device::cuda, linear_op, blas::cublas<float>, preconditioner> cg{linear_op{mat}, blas_t{},
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


void work_cuda_no_prec(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto h_mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto d_mat_buf = h_mat_buf.to<device::cuda>();
  auto mat = d_mat_buf.const_view();
  auto rows = mat.rows();
  auto nnz = mat.nnz();

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  auto d_csr_values = make_cuda_buffer<float>(nnz);
  auto d_csr_col_idx = make_cuda_buffer<index_t>(nnz);
  auto d_csr_row_ptr = make_cuda_buffer<index_t>(rows + 1);

  copy(d_csr_values.view(), values);
  copy(d_csr_col_idx.view(), col_idx);
  copy(d_csr_row_ptr.view(), row_ptr);

  using linear_op = sparse::iterative::sparse_matrix<sparse::blas::cusparse<float, sparse::sparse_format::csr>>;
  using blas_t = blas::cublas<float>;
  sparse::iterative::cg<float, device::cuda, linear_op, blas::cublas<float>> cg{linear_op{mat}, blas_t{}};

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

void work_cuda_ilu0(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto h_mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto d_mat_buf = h_mat_buf.to<device::cuda>();
  auto mat = d_mat_buf.const_view();
  auto rows = mat.rows();
  auto nnz = mat.nnz();

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  auto d_csr_values = make_cuda_buffer<float>(nnz);
  auto d_csr_col_idx = make_cuda_buffer<index_t>(nnz);
  auto d_csr_row_ptr = make_cuda_buffer<index_t>(rows + 1);

  copy(d_csr_values.view(), values);
  copy(d_csr_col_idx.view(), col_idx);
  copy(d_csr_row_ptr.view(), row_ptr);

  using linear_op = sparse::iterative::sparse_matrix<
      sparse::blas::cusparse<float, sparse::sparse_format::csr>>;
  using blas_t = blas::cublas<float>;
  using preconditioner =
      sparse::iterative::ilu<float, device::cuda, sparse::sparse_format::csr>;
  sparse::iterative::cg<float, device::cuda, linear_op, blas::cublas<float>,
                       preconditioner>
      cg{linear_op{mat}, blas_t{}, preconditioner{mat}};
  cg.preconditioner().compute();

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


void work_cuda_ic(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto h_mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto d_mat_buf = h_mat_buf.to<device::cuda>();
  auto mat = d_mat_buf.const_view();
  auto rows = mat.rows();
  auto nnz = mat.nnz();

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  auto d_csr_values = make_cuda_buffer<float>(nnz);
  auto d_csr_col_idx = make_cuda_buffer<index_t>(nnz);
  auto d_csr_row_ptr = make_cuda_buffer<index_t>(rows + 1);

  copy(d_csr_values.view(), values);
  copy(d_csr_col_idx.view(), col_idx);
  copy(d_csr_row_ptr.view(), row_ptr);

  using linear_op = sparse::iterative::sparse_matrix<
      sparse::blas::cusparse<float, sparse::sparse_format::csr>>;
  using blas_t = blas::cublas<float>;
  using preconditioner =
      sparse::iterative::ichol<float, device::cuda, sparse::sparse_format::csr>;
  sparse::iterative::cg<float, device::cuda, linear_op, blas::cublas<float>,
                       preconditioner>
      cg{linear_op{mat}, blas_t{}, preconditioner{mat}};
  cg.preconditioner().compute();

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


void work_cuda_ai(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto h_mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto d_mat_buf = h_mat_buf.to<device::cuda>();
  auto mat = d_mat_buf.const_view();
  auto rows = mat.rows();
  auto nnz = mat.nnz();

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  auto d_csr_values = make_cuda_buffer<float>(nnz);
  auto d_csr_col_idx = make_cuda_buffer<index_t>(nnz);
  auto d_csr_row_ptr = make_cuda_buffer<index_t>(rows + 1);

  copy(d_csr_values.view(), values);
  copy(d_csr_col_idx.view(), col_idx);
  copy(d_csr_row_ptr.view(), row_ptr);

  using linear_op = sparse::iterative::sparse_matrix<
      sparse::blas::cusparse<float, sparse::sparse_format::csr>>;
  using blas_t = blas::cublas<float>;
  using preconditioner = sparse::iterative::approx_inverse_preconditioner<
      float, device::cuda, sparse::sparse_format::csr,
      sparse::blas::cusparse<float, mathprim::sparse::sparse_format::csr>>;
  sparse::iterative::cg<float, device::cuda, linear_op, blas::cublas<float>,
                       preconditioner>
      cg{linear_op{mat}, blas_t{}, preconditioner{mat}};
  cg.preconditioner().compute();

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

// BENCHMARK(work_eigen_naive)->Range(1 << 10, 1 << 16);
// BENCHMARK_TEMPLATE(work, blas::cpu_handmade<float>)->Range(1 << 10, 1 << 16);
// BENCHMARK_TEMPLATE(work, blas::cpu_eigen<float>)->Range(1 << 10, 1 << 16);
// BENCHMARK(work_chol)->Range(1 << 10, 1 << 16);
#ifdef NDEBUG
#define LARGE_RANGE 1 << 10
#else
#define LARGE_RANGE 1 << 5
#endif
// BENCHMARK_TEMPLATE(work_ic, blas::cpu_eigen<float>)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);
// BENCHMARK_TEMPLATE(work, blas::cpu_eigen<float>)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);
BENCHMARK(work_cuda)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);
BENCHMARK(work_cuda_no_prec)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);
BENCHMARK(work_cuda_ilu0)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);
BENCHMARK(work_cuda_ic)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);
BENCHMARK(work_cuda_ai)->Range(1 << 4, LARGE_RANGE)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();