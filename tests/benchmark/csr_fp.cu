#include <benchmark/benchmark.h>

#include <Eigen/Sparse>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>

#include "mathprim/blas/cublas.cuh"
#include "mathprim/linalg/iterative/smoother/fixed_iteration.hpp"
#include "mathprim/linalg/iterative/smoother/jacobi.hpp"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/sparse/blas/cusparse.hpp"
#include "mathprim/sparse/systems/laplace.hpp"

using namespace mathprim;

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

  using linear_op = sparse::blas::cusparse<float, sparse::sparse_format::csr>;
  using blas_t = blas::cublas<float>;
  using smoother_t = sparse::iterative::jacobi_smoother<float, device::cuda, sparse::sparse_format::csr, blas_t>;
  sparse::iterative::basic_fixed_iteration<float, device::cuda, mathprim::sparse::sparse_format::csr, linear_op, blas_t,
                                           smoother_t>
      jacobi{mat};

  auto d_b = make_cuda_buffer<float>(rows);
  auto d_x = make_cuda_buffer<float>(rows);
  auto parfor = par::cuda();
  for (auto _ : state) {
    state.PauseTiming();
    parfor.run(make_shape(rows), [d_xv = d_x.view()] __device__(index_t i) {
      d_xv[i] = 1.0f;
    });
    // b = A * x
    jacobi.linear_operator().gemv(1.0f, d_x.view(), 0.0f, d_b.view());

    parfor.run(make_shape(rows), [d_xv = d_x.view(), d_bv = d_b.view()] __device__(index_t i) {
      d_xv[i] = (i % 100 - 50) / 100.0f;
    });
    parfor.sync();
    state.ResumeTiming();

    // auto result = jacobi.apply(
    //     d_b.view(), d_x.view(),
    //     {
    //         .max_iterations_ = dsize * 4,
    //         .norm_tol_ = 1e-6f,
    //     } //  , [](auto iter, auto norm) {
    //       //   std::cout << "Iter: " << iter << " Norm: " << norm << std::endl;
    //       //  }
    // );
    auto result = jacobi.solve(d_x.view(), d_b.view(), {dsize * 4, 1e-6f});
    parfor.sync();
    state.SetLabel(std::to_string(result.norm_));
  }
}

#ifdef NDEBUG
#define LARGE_RANGE 1 << 10
#else
#define LARGE_RANGE 1 << 5
#endif
BENCHMARK(work_cuda)->Range(1 << 2, 1 << 5)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();