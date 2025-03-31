#include <fstream>
#include <iostream>
#include "mathprim/blas/cublas.cuh"
#include "mathprim/linalg/iterative/precond/fsai0.hpp"
#include "mathprim/linalg/iterative/precond/diagonal.hpp"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/sparse/blas/cusparse.hpp"
#include "mathprim/sparse/cvt.hpp"
#include "mathprim/linalg/direct/cholmod.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/supports/io/matrix_market.hpp"
#include "mathprim/supports/stringify.hpp"

// Reading matrix from file
// Loader time: 10149 ms
// Matrix size: (3609455, 3609455)
// Cholesky decomposition time: 28912 ms
// Solve CHOLMOD time: 331 ms
// Transfer GPU time: 104 ms
// FSAI precompute time: 395 ms
// Iter 0 norm 13.1403
// Iter 1000 norm 0.277579
// Iter 2000 norm 0.0178386
// Iter 3000 norm 0.000593978
// Iter 4000 norm 1.73722e-05
// Solve CG time: 17025 ms

int main() {
  using namespace mathprim;
  io::matrix_market<float> mm;
  sparse::basic_sparse_matrix<float, device::cpu, sparse::sparse_format::csr> matrix_cpu;

  auto start = std::chrono::high_resolution_clock::now();
  auto report_time = [&start] (const std::string& msg) {
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << msg << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    start = end;
  };
  {
    std::cout << "Reading matrix from file" << std::endl;
    auto file = std::ifstream("laplacian.mtx");
    auto matrix_cpu_coo = mm.read(file);
    matrix_cpu = sparse::format_convert<float, sparse::sparse_format::csr, device::cpu>::from_coo(
        matrix_cpu_coo.const_view());
    visit(matrix_cpu.view(), par::seq{}, [](index_t i, index_t j, float& val) {
      val = -val;
      if (i == j) {
        val += 1e-4;
      }
    });
  }
  report_time("Loader");
  std::cout << "Matrix size: " << matrix_cpu.view().shape() << std::endl;
  int n = matrix_cpu.rows();

  sparse::direct::cholmod_chol<float, mathprim::sparse::sparse_format::csr> csr_super(matrix_cpu.const_view());
  report_time("Cholesky decomposition");
  auto x = make_buffer<float>(n), b = make_buffer<float>(n);
  auto xv = x.view(), bv = b.view();
  for (int i = 0; i < n; ++i) {
    xv[i] = 1.0;
    bv[i] = 0.0;
  }

  start = std::chrono::high_resolution_clock::now();
  csr_super.solve(bv, xv);
  report_time("Solve CHOLMOD");

  // GPU.
  auto gpu = matrix_cpu.to<device::cuda>();
  using sp_blas_t = sparse::blas::cusparse<float, mathprim::sparse::sparse_format::csr>;
  using blas_t = blas::cublas<float>;
  using prec_t = sparse::iterative::fsai0_preconditioner<sp_blas_t>;
  // using prec_t = sparse::iterative::diagonal_preconditioner<
  //     float, device::cuda, sparse::sparse_format::csr, blas_t>;
  report_time("Transfer GPU");
  sparse::iterative::cg<float, device::cuda, sp_blas_t, blas_t, prec_t> cg(gpu.const_view());
  report_time("FSAI precompute");
  auto d_x = make_cuda_buffer<float>(n), d_b = make_cuda_buffer<float>(n);
  auto d_xv = d_x.view(), d_bv = d_b.view();

  par::cuda().run(make_shape(n), [d_xv, d_bv] __device__ (int i) {
    d_xv[i] = 0.0;
    d_bv[i] = 1.0;
  });

  start = std::chrono::high_resolution_clock::now();
  cg.solve(d_xv, d_bv, {n * 4, 1e-6}, [](index_t iter, float norm) {
    if (iter % 1000 == 0) {
      std::cout << "Iter " << iter << " norm " << norm << std::endl;
    }
  });
  report_time("Solve CG");

  return 0;
}