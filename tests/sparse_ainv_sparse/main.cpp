#include "mathprim/linalg/iterative/precond/sparse_inverse.hpp"
#include <fstream>
#include <Eigen/Sparse>
#include <iostream>
#include <mathprim/blas/cpu_blas.hpp>
#include <mathprim/blas/cpu_eigen.hpp>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/cholmod.hpp>
#include <mathprim/sparse/cvt.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/sparse/systems/laplace.hpp>

#include <mathprim/supports/io/matrix_market.hpp>

int main () {
  using namespace mathprim;

  using spblas_t = sparse::blas::eigen<float, sparse::sparse_format::csr>;
  using blas_t = blas::cpu_blas<float>;
  using prec_t = sparse::iterative::sparse_preconditioner<spblas_t, blas_t>;
  using ainv_t = sparse::iterative::approx_inverse_preconditioner<spblas_t>;
  using cg_t = sparse::iterative::cg<float, device::cpu, spblas_t, blas_t, prec_t>;
  index_t N = 128;
  auto matrix = sparse::laplace_operator<float, 2>(make_shape(N, N)).matrix<mathprim::sparse::sparse_format::csr>();
  ainv_t ainv(matrix.const_view());

  cg_t solver(matrix.const_view());
  solver.preconditioner().derived().set_approximation(ainv.ainv(), 1e-4);

  auto b = make_buffer<float, device::cpu>(N * N);
  auto x = make_buffer<float, device::cpu>(N * N);

  x.fill_bytes(0);
  par::seq().run(make_shape(N * N), [&](index_t i) {
    x.view()[i] = 1;
  });
  solver.linear_operator().gemv(1., x.view(), 0., b.view()); // b = A * x
  par::seq().run(make_shape(N * N), [&](index_t i) {
    x.view()[i] = 0;
  });

  auto result = solver.solve(x.view(), b.const_view(), {10240, 1e-4});

  printf("After %d iterations, got residual %f\n", result.iterations_, result.norm_);
  return 0;
}