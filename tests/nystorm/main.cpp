#include <benchmark/benchmark.h>

#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/linalg/iterative/precond/diagonal.hpp"
#include "mathprim/linalg/iterative/precond/low_rank_approximation.hpp"
#include "mathprim/linalg/iterative/solver/cg.hpp"
#include "mathprim/parallel/openmp.hpp"
#include "mathprim/sparse/blas/naive.hpp"
#include "mathprim/sparse/systems/laplace.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;
using blas_t = blas::cpu_blas<float>;
using low_rank = sparse::iterative::low_rank_preconditioner<float, device::cpu, sparse::sparse_format::csr, blas_t>;
using linear_op = sparse::iterative::sparse_matrix<sparse::blas::naive<float, sparse::sparse_format::csr, par::seq>>;
using none = sparse::iterative::none_preconditioner<float, device::cpu>;

template <typename Blas>
void do_test_exact(benchmark::State& state) {
  using solver = sparse::iterative::cg<float, device::cpu, linear_op, blas_t, low_rank>;
  index_t dsize = state.range(0);
  auto laplacian = sparse::laplace_operator<float, 2>(make_shape(dsize, dsize)).matrix<sparse::sparse_format::csr>();
  index_t n = laplacian.rows();
  index_t k = state.range(1);
  auto dense = eigen_support::map(laplacian.const_view()).toDense();
  auto eigsh = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf>(dense);
  auto smallest_eigenvectors = eigsh.eigenvectors().leftCols(k).eval();  // (n, k)
  auto smallest_eigenvalues = eigsh.eigenvalues().head(k).eval();
  {
    float largest = smallest_eigenvalues.maxCoeff();
    smallest_eigenvalues.array() = 1.0f / smallest_eigenvalues.array();  // inv(D)
    smallest_eigenvalues.array() *= largest;                             // inv(D) / s
    smallest_eigenvalues.array() -= 1.0f;     // inv(D) / s - Id.
  }

  solver cg(linear_op(laplacian.const_view()), blas_t(), low_rank{n, k});

  copy(cg.preconditioner().basis().view(), eigen_support::view(smallest_eigenvectors));
  copy(cg.preconditioner().diag().view(), eigen_support::view(smallest_eigenvalues));

  auto b = make_buffer<float>(n);
  auto x = make_buffer<float>(n);
  auto x_view = x.view();

  for (auto _ : state) {
    par::seq().run(make_shape(n), [x_view](index_t i) {
      x_view[i] = 1.0f;
    });
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());
    par::seq().run(make_shape(n), [x_view](index_t i) {
      x_view[i] = 0.0f;
    });
    auto result = cg.apply(b.const_view(), x.view(), {n, 1e-7f});
    state.SetLabel(std::to_string(result.iterations_) + ": " + std::to_string(result.norm_ / 1e-7f));
  }
}

template <typename Blas>
void do_test_diagonal(benchmark::State& state) {
  using precond = sparse::iterative::diagonal_preconditioner<float, device::cpu, sparse::sparse_format::csr, blas_t>;
  using solver = sparse::iterative::cg<float, device::cpu, linear_op, Blas, precond>;
  index_t dsize = state.range(0);
  auto laplacian = sparse::laplace_operator<float, 2>(make_shape(dsize, dsize)).matrix<sparse::sparse_format::csr>();
  index_t n = laplacian.rows();

  solver cg(linear_op(laplacian.const_view()), blas_t(), precond(laplacian.const_view()));

  auto b = make_buffer<float>(n);
  auto x = make_buffer<float>(n);
  auto x_view = x.view();

  for (auto _ : state) {
    par::seq().run(make_shape(n), [x_view](index_t i) {
      x_view[i] = 1.0f;
    });
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());
    par::seq().run(make_shape(n), [x_view](index_t i) {
      x_view[i] = 0.0f;
    });
    auto result = cg.apply(b.const_view(), x.view(), {n, 1e-7f});
    state.SetLabel(std::to_string(result.iterations_) + ": " + std::to_string(result.norm_ / 1e-7f));
  }
}

// Use a analytical solution for the basis:
//  b_{i,j} (x, y) = sin(i * pi * x) * sin(j * pi * y)
float basis_fn(index_t i, index_t j, index_t dsize, index_t x, index_t y) {
  auto pi = 3.14159265358979323846f;
  auto xx = static_cast<float>(x) / (dsize - 1);
  auto yy = static_cast<float>(y) / (dsize - 1);
  return sin((i + 1) * pi * xx) * sin((j + 1) * pi * yy);
}

void do_test_ana(benchmark::State& state) {
  using solver = sparse::iterative::cg<float, device::cpu, linear_op, blas_t, low_rank>;
  index_t dsize = state.range(0);
  auto laplacian = sparse::laplace_operator<float, 2>(make_shape(dsize, dsize)).matrix<sparse::sparse_format::csr>();
  index_t n = laplacian.rows();
  index_t k = state.range(1);

  solver cg(linear_op(laplacian.const_view()), blas_t(), low_rank{n, k});

  {
    auto kk = static_cast<index_t>(round(sqrt(k)));
    auto temp_buffer = make_buffer<float>(dsize, dsize);
    auto temp_buffer2 = make_buffer<float>(dsize, dsize);
    auto temp = temp_buffer.view(), temp2 = temp_buffer2.view();
    auto basis = cg.preconditioner().basis().view();
    auto diag = cg.preconditioner().diag().view();
    blas_t bl;
    for (index_t i = 0; i < kk; ++i) {
      for (index_t j = 0; j < kk; ++j) {
        auto dims = sparse::laplace_operator<float, 2>(make_shape(dsize, dsize)).dims();
        for (auto coord : dims) {
          temp(coord) = basis_fn(i, j, dsize, coord[0], coord[1]);
        }

        // Unitize the basis.
        float norm = bl.norm(temp.as_const());
        float scale = 1.0f / norm;
        bl.scal(scale, temp);

        // Copy the basis to the buffer.
        copy(basis.slice(i * kk + j), temp.flatten());

        // the diagonal is the eigenvalue.
        cg.linear_operator().apply(1.0f, temp.flatten(), 0.0f, temp2.flatten()); // temp2 = A * temp

        float dot = bl.dot(temp.as_const(), temp2.as_const());
        diag[i * kk + j] = dot;
      }
    }

    // modify the diagonal buffer to (inv(D)/s - Id)
    float largest = diag[bl.amax(diag.as_const())];
    for (index_t i = 0; i < k; ++i) {
      diag[i] = largest / diag[i] - 1.0f;
      // printf("%d:%.4e\n", i, diag[i]);
    }
  }

  auto b = make_buffer<float>(n);
  auto x = make_buffer<float>(n);
  auto x_view = x.view();
  for (auto _ : state) {
    par::seq().run(make_shape(n), [x_view](index_t i) {
      x_view[i] = 1.0f;
    });
    cg.linear_operator().apply(1.0f, x.view(), 0.0f, b.view());
    par::seq().run(make_shape(n), [x_view](index_t i) {
      x_view[i] = 0.0f;
    });
    auto result = cg.apply(b.const_view(), x.view(), {n, 1e-7f});
    state.SetLabel(std::to_string(result.iterations_) + ": " + std::to_string(result.norm_ / 1e-7f));
  }
}
std::vector<int64_t> range_dsize{16, 24, 32, 64, 128, 256, 1024};
BENCHMARK(do_test_ana)->ArgsProduct({range_dsize, {4, 16, 64, 256}})->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(do_test_exact, blas_t)->ArgsProduct({{16, 24, 32}, {4, 16, 64, 256}})->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(do_test_diagonal, blas_t)->ArgsProduct({range_dsize})->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
