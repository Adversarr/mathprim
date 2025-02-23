#include <benchmark/benchmark.h>

#include <mathprim/core/buffer.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/sparse/blas/eigen.hpp>

using namespace mathprim;

template <typename be, bool trans>
void csr(benchmark::State& state) {
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
      values[i * 3] = 2.0f;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      values.as_const(), row_ptr.as_const(), col_idx.as_const(), rows, cols, nnz, sparse::sparse_property::general);

  auto h_x = make_buffer<float>(cols);
  auto h_y = make_buffer<float>(rows);
  sparse::blas::naive<float, sparse::sparse_format::csr, be> blas(mat);
  for (auto _ : state) {
    blas.gemv(1.0f, h_x.const_view(), 0.0f, h_y.view(), trans);
  }
}


template <bool trans>
void eigen_sparse_map(benchmark::State& state) {
  Eigen::initParallel();
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
      values[i * 3] = 2.0f;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      values.as_const(), row_ptr.as_const(), col_idx.as_const(), rows, cols, nnz, sparse::sparse_property::general);

  auto h_x = make_buffer<float>(cols);
  auto h_y = make_buffer<float>(rows);
  sparse::blas::eigen<float, sparse::sparse_format::csr> blas(mat);
  for (auto _ : state) {
    blas.gemv(1.0f, h_x.const_view(), 0.0f, h_y.view(), trans);
  }
}




BENCHMARK_TEMPLATE(csr, par::seq, false)->Range(1 << 10, 1 << 18)->RangeMultiplier(2);
BENCHMARK_TEMPLATE(csr, par::openmp, false)->Range(1 << 10, 1 << 18)->RangeMultiplier(2);
BENCHMARK_TEMPLATE(eigen_sparse_map, false)->Range(1 << 10, 1 << 18)->RangeMultiplier(2);

BENCHMARK_TEMPLATE(csr, par::seq, true)->Range(1 << 10, 1 << 18)->RangeMultiplier(2);
BENCHMARK_TEMPLATE(eigen_sparse_map, true)->Range(1 << 10, 1 << 18)->RangeMultiplier(2);

BENCHMARK_MAIN();