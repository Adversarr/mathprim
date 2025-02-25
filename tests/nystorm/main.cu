#include <benchmark/benchmark.h>
#include <iostream>
#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/linalg/iterative/precond/diagonal.hpp"
#include "mathprim/linalg/iterative/precond/low_rank_approximation.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/sparse/blas/cusparse.hpp"
#include "mathprim/sparse/blas/naive.hpp"
#include "mathprim/sparse/systems/laplace.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;
using blas_t = blas::cublas<float>;
using low_rank = iterative_solver::low_rank_preconditioner<
    float, device::cuda, sparse::sparse_format::csr, blas_t>;
using linear_op = iterative_solver::sparse_matrix<
    sparse::blas::cusparse<float, sparse::sparse_format::csr>>;

// Use a analytical solution for the basis:
//  b_{i,j} (x, y) = sin(i * pi * x) * sin(j * pi * y)
float basis_fn(index_t i, index_t j, index_t dsize, index_t x, index_t y) {
  auto pi = M_PI;
  auto xx = static_cast<float>(x) / (dsize - 1);
  auto yy = static_cast<float>(y) / (dsize - 1);
  return sin((i + 1) * pi * xx) * sin((j + 1) * pi * yy);
}

template <typename Blas> void do_test_exact(benchmark::State &state) {
  using solver =
      iterative_solver::cg<float, device::cuda, linear_op, blas_t, low_rank>;
  index_t dsize = state.range(0);
  auto laplacian = sparse::laplace_operator<float, 2>(make_shape(dsize, dsize))
                       .matrix<sparse::sparse_format::csr>();
  index_t n = laplacian.rows();
  index_t k = state.range(1);
  auto dense = eigen_support::map(laplacian.const_view()).toDense();
  auto eigsh = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf>(dense);
  auto smallest_eigenvectors =
      eigsh.eigenvectors().leftCols(k).eval(); // (n, k)
  auto smallest_eigenvalues = eigsh.eigenvalues().head(k).eval();
  {
    float largest = smallest_eigenvalues.maxCoeff();
    smallest_eigenvalues = smallest_eigenvalues.cwiseInverse(); // inv(D)
    smallest_eigenvalues.array() *= largest;                    // inv(D) / s
    smallest_eigenvalues.array() -= 1.0f; // inv(D) / s - Id.
  }
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    // compare the eigenvalues with basis.
    auto temp = make_buffer<float>(dsize, dsize);
    auto temp_view = temp.view();
    auto kk = static_cast<index_t>(round(sqrt(k)));
    blas::cpu_blas<float> h_bl;

    for (index_t i = 0; i < kk; ++i) {
      for (index_t j = 0; j < kk; ++j) {
        for (auto coord : make_shape(dsize, dsize)) {
          temp_view(coord) = basis_fn(i, j, dsize, coord[0], coord[1]);
        }

        // normalize it
        float norm = h_bl.norm(temp_view.as_const());
        float scale = 1.0f / norm;
        h_bl.scal(scale, temp_view);

        // compare with smallest eigen values and find the dot product's
        // largest.
        auto temp_map = eigen_support::cmap(temp_view.flatten());
        float largest = 0, largest_norm = 0;
        index_t amax = 0;
        for (index_t l = 0; l < k; ++l) {
          float dot = temp_map.dot(smallest_eigenvectors.col(l));
          if (dot > largest) {
            largest = dot;
            largest_norm = smallest_eigenvectors.col(l).norm();
            amax = l;
          }
        }
        printf("(%d, %d): %d %.4e %.4e %.4e\n", i, j, amax, temp_map.norm(),
               largest_norm, largest);
      }
    }
  });

  auto d_l = laplacian.to(device::cuda{});
  solver cg(linear_op(d_l.const_view()), blas_t(), low_rank{n, k});

  copy(cg.preconditioner().basis().view(),
       eigen_support::view(smallest_eigenvectors));
  copy(cg.preconditioner().diag().view(),
       eigen_support::view(smallest_eigenvalues));

  auto b = make_cuda_buffer<float>(n);
  auto x = make_cuda_buffer<float>(n);
  auto x_view = x.view();

  for (auto _ : state) {
    state.PauseTiming();
    par::cuda().run(make_shape(n),
                    [x_view] __device__(index_t i) { x_view[i] = 1.0f; });
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());
    x.fill_bytes(0);
    cudaDeviceSynchronize();
    state.ResumeTiming();
    auto result = cg.apply(b.const_view(), x.view(), {n, 1e-7f});
    state.SetLabel(std::to_string(result.iterations_) + ": " +
                   std::to_string(result.norm_ / 1e-7f));
  }
}

template <typename Blas> void do_test_diagonal(benchmark::State &state) {
  using precond = mathprim::iterative_solver::diagonal_preconditioner<
      float, device::cuda, sparse::sparse_format::csr, Blas>;
  using solver =
      iterative_solver::cg<float, device::cuda, linear_op, blas_t, precond>;
  index_t ndim = state.range(0);
  auto laplacian = sparse::laplace_operator<float, 2>(make_shape(ndim, ndim))
                       .matrix<sparse::sparse_format::csr>();
  index_t n = laplacian.rows();
  auto d_l = laplacian.to(device::cuda{});
  solver cg(linear_op(d_l.const_view()), blas_t{}, precond{d_l.const_view()});

  auto b = make_cuda_buffer<float>(n);
  auto x = make_cuda_buffer<float>(n);
  auto x_view = x.view();

  for (auto _ : state) {
    par::cuda().run(make_shape(n),
                    [x_view] __device__(index_t i) { x_view[i] = 1.0f; });
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());
    x.fill_bytes(0);

    auto result = cg.apply(b.const_view(), x.view(), {n, 1e-7f});
    state.SetLabel(std::to_string(result.iterations_) + ": " +
                   std::to_string(result.norm_ / 1e-7f));
  }
}

void do_test_ana(benchmark::State &state) {
  using solver =
      iterative_solver::cg<float, device::cuda, linear_op, blas_t, low_rank>;
  index_t dsize = state.range(0);
  auto laplacian = sparse::laplace_operator<float, 2>(make_shape(dsize, dsize))
                       .matrix<sparse::sparse_format::csr>();
  index_t n = laplacian.rows();
  index_t k = state.range(1);

  auto d_l = laplacian.to(device::cuda{});
  solver cg(linear_op(d_l.const_view()), blas_t(), low_rank{n, k});

  {
    auto kk = static_cast<index_t>(round(sqrt(k)));
    auto temp_buffer = make_buffer<float>(dsize, dsize);
    auto temp_buffer2 = make_cuda_buffer<float>(dsize, dsize);
    auto temp = temp_buffer.view();
    auto temp2 = temp_buffer2.view();
    auto h_diag = make_buffer<float>(k);
    auto basis = cg.preconditioner().basis().view();
    auto h_diag_view = h_diag.view();
    blas_t d_bl;
    blas::cpu_blas<float> h_bl;
    for (index_t i = 0; i < kk; ++i) {
      for (index_t j = 0; j < kk; ++j) {
        auto dims =
            sparse::laplace_operator<float, 2>(make_shape(dsize, dsize)).dims();
        for (auto coord : dims) {
          temp(coord) = basis_fn(i, j, dsize, coord[0], coord[1]);
        }

        // Unitize the basis.
        float norm = h_bl.norm(temp.as_const());
        float scale = 1.0f / norm;
        h_bl.scal(scale, temp);

        // Copy the basis to the buffer.
        copy(basis.slice(i * kk + j), temp.flatten());

        // the diagonal is the eigenvalue.
        cg.linear_operator().apply(1.0f, basis.slice(i * kk + j), 0.0f,
                          temp2.flatten()); // temp2 = A * temp

        float dot = d_bl.dot(basis.slice(i * kk + j).as_const(),
                             temp2.flatten().as_const());
        h_diag_view[i * kk + j] = dot;
      }
    }

    // modify the diagonal buffer to (inv(D)/s - Id)
    float largest =
        h_diag_view[blas::cpu_blas<float>{}.amax(h_diag_view.as_const())];
    for (index_t i = 0; i < k; ++i) {
      h_diag_view[i] = largest / h_diag_view[i] - 1.0f;
    }
    // copy the diagonal buffer to the preconditioner.
    copy(cg.preconditioner().diag().view(), h_diag_view);
  }
  auto b = make_cuda_buffer<float>(n);
  auto x = make_cuda_buffer<float>(n);
  auto x_view = x.view();

  for (auto _ : state) {
    par::cuda().run(make_shape(n),
                    [x_view] __device__(index_t i) { x_view[i] = 1.0f; });
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());
    x.fill_bytes(0);
    auto result = cg.apply(b.const_view(), x.view(), {n, 1e-7f});

    state.SetLabel(std::to_string(result.iterations_) + ": " +
                   std::to_string(result.norm_ / 1e-7f));
  }
}

std::vector<int64_t> range_dsize{1024};
std::vector<int64_t> range_k{16, 64, 256};
// std::vector<int64_t> range_dsize{256};
// std::vector<int64_t> range_k{64, 256};
BENCHMARK(do_test_ana)
    ->ArgsProduct({range_dsize, range_k})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(do_test_exact, blas_t)
    ->ArgsProduct({{32}, {16}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(do_test_diagonal, blas_t)
    ->ArgsProduct({range_dsize})
    ->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
