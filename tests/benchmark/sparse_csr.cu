#include <benchmark/benchmark.h>

#include <mathprim/core/buffer.hpp>
#include <mathprim/parallel/openmp.hpp>
#include "mathprim/blas/cublas.cuh"
#include "mathprim/sparse/blas/cusparse.hpp"
#include <mathprim/sparse/blas/eigen.hpp>
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

  using blas_t = sparse::blas::cusparse<float, mathprim::sparse::sparse_format::csr>;
  blas_t bl(mat);

  auto x = make_cuda_buffer<float>(rows);
  auto y = make_cuda_buffer<float>(rows);

  for (auto _ : state) {
    bl.gemv(1.0, x.const_view(), 0.0, y.view());
    cudaDeviceSynchronize();
  }
}
void work_cuda_spmm(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto h_mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto d_mat_buf = h_mat_buf.to<device::cuda>();
  auto mat = d_mat_buf.const_view();
  auto rows = mat.rows();
  auto nnz = mat.nnz();

  using blas_t = sparse::blas::cusparse<float, mathprim::sparse::sparse_format::csr>;
  blas_t bl2(mat);
  blas_t bl;
  bl = std::move(bl2);

  auto x = make_cuda_buffer<float>(rows, 32);
  auto y = make_cuda_buffer<float>(rows, 32);

  for (auto _ : state) {
    bl.spmm(1.0, x.const_view(), 0.0, y.view());
    cudaDeviceSynchronize();
  }
}


BENCHMARK(work_cuda)->ArgsProduct({{512, 1024}});
BENCHMARK(work_cuda_spmm)->ArgsProduct({{512, 1024}});

BENCHMARK_MAIN();