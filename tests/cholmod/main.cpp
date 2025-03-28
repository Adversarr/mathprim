#include "benchmark/benchmark.h"
#include <iostream>

#include "mathprim/linalg/direct/cholmod.hpp"
#include "mathprim/linalg/direct/eigen_support.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/blas/naive.hpp"
#include "mathprim/sparse/systems/laplace.hpp"
#include "mathprim/supports/eigen_sparse.hpp"
#include "mathprim/supports/stringify.hpp"
using namespace mathprim;

template <typename Scalar, typename Solver>
void work(benchmark::State &state) {
  int dsize = state.range(0);
  sparse::laplace_operator<Scalar, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.template matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();
  auto b = make_buffer<Scalar>(rows);
  auto x = make_buffer<Scalar>(rows);

  auto vb = make_buffer<Scalar>(rows, 5);
  auto vx = make_buffer<Scalar>(rows, 5);
  sparse::visit(mat_buf.view(), par::seq(), [&](index_t i, index_t j, auto &v) {
    if (i == j) {
      v += 1;
    }
  });

  sparse::blas::naive<Scalar, sparse::sparse_format::csr, par::seq> bl{mat};
  // GT = ones.
  par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
    xv[i] = 1.0f;
  });
  par::seq().run(make_shape(rows, 5), [vxv = vx.view()](auto ij) {
    auto [i, j] = ij;
    vxv(i, j) = 1.0f;
  });

  // b = A * x
  bl.gemv(1.0f, x.view(), 0.0f, b.view());
  bl.spmm(1.0f, vx.view(), 0.0f, vb.view());

  // x = (i % 100 - 50) / 100.0f
  par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
    xv[i] = (i % 100 - 50) / 100.0f;
  });
  par::seq().run(make_shape(rows, 5), [vxv = vx.view()](auto ij) {
    auto [i, j] = ij;
    vxv(i, j) = (i % 100 - 50) / 100.0f;
  });

  // solve
  // auto chol = sparse::direct::cholmod_chol<Scalar, device::cpu>{mat};
  auto start_precond = std::chrono::high_resolution_clock::now();
  auto chol = Solver{mat};
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start_precond);
  std::cout << "Preconditioner time: " << duration.count() << " ms" << std::endl;
  for (auto _ : state) {
    x.fill_bytes(0);
    vx.fill_bytes(0);
    chol.solve(x.view(), b.view()); // A x = b
    chol.vsolve(vx.view(), vb.view());
  }
  // checking
  auto b2 = make_buffer<Scalar>(rows);
  bl.gemv(1.0f, x.view(), 0.0f, b2.view());
  auto bv = b.view(), b2v = b2.view();
  for (index_t i = 0; i < rows; ++i) {
    if (std::abs(bv[i] - b2v[i]) > 1e-4f) {
      std::cerr << "Error at " << i << " " << bv[i] << " " << b2v[i] << std::endl;
      state.SkipWithError("Error");
      break;
    }
  }
}

BENCHMARK_TEMPLATE(work, double, sparse::direct::eigen_simplicial_chol<double, sparse::sparse_format::csr>)
    ->Range(16, 512)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(work, double, sparse::direct::eigen_cholmod_simplicial_ldlt<sparse::sparse_format::csr>)
    ->Range(16, 1024)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(work, double, sparse::direct::eigen_cholmod_simplicial_llt<sparse::sparse_format::csr>)
    ->Range(16, 1024)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(work, float, sparse::direct::cholmod_chol<float, sparse::sparse_format::csr>)
    ->Range(16, 512)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(work, double, sparse::direct::cholmod_chol<double, sparse::sparse_format::csr>)
    ->ArgsProduct({{512, 1024, 2048}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(work, double, sparse::direct::eigen_cholmod_supernodal_llt<sparse::sparse_format::csr>)
    ->ArgsProduct({{512, 1024, 2048}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();