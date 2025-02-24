#include <benchmark/benchmark.h>

#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cpu_handmade.hpp"
#include "mathprim/core/buffer.hpp"

using namespace mathprim;

static void BM_axpy_handmade(benchmark::State& state) {
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  blas::cpu_handmade<float> b;
  for (auto _ : state) {
    b.axpy(1.0f, x.view().as_const(), y.view());
  }
}

static void BM_axpy_eigen(benchmark::State& state) {
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  blas::cpu_eigen<float> b;
  for (auto _ : state) {
    b.axpy(1.0f, x.const_view(), y.view());
  }
}

static void BM_axpy_blas(benchmark::State& state) {
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));

  blas::cpu_blas<float> b;
  for (auto _ : state) {
    b.axpy(1.0f, x.view().as_const(), y.view());
  }
}

static void BM_gemv_handmade(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto x_view = x.view().as_const();
  auto y_view = y.view();
  blas::cpu_handmade<float> b;
  for (auto _ : state) {
    b.gemv(1.0f, a_view, x_view, 1.0f, y_view);
  }
}

static void BM_gemm_handmade(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto B = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto C = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto b_view = B.view().as_const();
  auto c_view = C.view();

  blas::cpu_handmade<float> b;
  for (auto _ : state) {
    b.gemm(1.0f, a_view, b_view, 1.0f, c_view);
  }
}

static void BM_gemv_eigen(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto x_view = x.view().as_const();
  auto y_view = y.view();
  blas::cpu_eigen<float> blas;
  for (auto _ : state) {
    blas.gemv(1.0f, a_view, x_view, 1.0f, y_view);
  }
}

static void BM_gemv_eigen_naive(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto x_view = x.view().as_const();
  auto y_view = y.view();
  for (auto _ : state) {
    // blas.gemv(1.0f, a_view, x_view, 1.0f, y_view);
    auto a = Eigen::Map<const Eigen::MatrixXf>(a_view.data(), a_view.shape(0), a_view.shape(1));
    auto x = Eigen::Map<const Eigen::VectorXf>(x_view.data(), x_view.shape(0));
    auto y = Eigen::Map<Eigen::VectorXf>(y_view.data(), y_view.shape(0));
    y *= 1.0f;
    y += a.transpose() * x;
  }
}

static void BM_gemm_eigen(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto B = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto C = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto b_view = B.view().as_const();
  auto c_view = C.view();
  blas::cpu_eigen<float> blas;
  for (auto _ : state) {
    blas.gemm(1.0f, a_view, b_view, 1.0f, c_view);
  }
}

static void BM_gemv_blas(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto x_view = x.view().as_const();
  auto y_view = y.view();
  blas::cpu_blas<float> b;
  for (auto _ : state) {
    b.gemv(1.0f, a_view, x_view, 1.0f, y_view);
  }
}

static void BM_gemm_blas(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto B = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto C = make_buffer<float>(static_cast<index_t>(state.range(0)), static_cast<index_t>(state.range(0)));
  auto a_view = A.view().as_const();
  auto b_view = B.view().as_const();
  auto c_view = C.view();

  blas::cpu_blas<float> b;
  for (auto _ : state) {
    b.gemm(1.0f, a_view, b_view, 1.0f, c_view);
  }
}

template <typename Blas, index_t N>
static void BM_batched_gemm(benchmark::State& state) {
  index_t batch_size = static_cast<index_t>(state.range(0));
  auto n = internal::holder<N>{};
  auto a = make_buffer<float>(batch_size, n, n);
  auto b = make_buffer<float>(batch_size, n, n);
  auto c = make_buffer<float>(batch_size, n, n);

  auto a_view = a.view().as_const();
  auto b_view = b.view().as_const();
  auto c_view = c.view();
  Blas bl;
  for (auto _ : state) {
    bl.gemm_batched(1.0f, a_view, b_view, 1.0f, c_view);
  }
}

BENCHMARK(BM_gemv_handmade)->Range(4, 4 << 10);
BENCHMARK(BM_gemv_eigen)->Range(4, 4 << 10);
BENCHMARK(BM_gemv_blas)->Range(4, 4 << 10);
BENCHMARK(BM_gemv_eigen_naive)->Range(4, 4 << 10);
BENCHMARK(BM_gemm_handmade)->Range(4, 4 << 7);
BENCHMARK(BM_gemm_eigen)->Range(4, 4 << 7);
BENCHMARK(BM_gemm_blas)->Range(4, 4 << 7);
BENCHMARK(BM_axpy_handmade)->Range(4, 4 << 14);
BENCHMARK(BM_axpy_eigen)->Range(4, 4 << 14);
BENCHMARK(BM_axpy_blas)->Range(4, 4 << 14);

BENCHMARK_TEMPLATE(BM_batched_gemm, blas::cpu_blas<float>, 4)->Range(4, 4 << 12);
BENCHMARK_TEMPLATE(BM_batched_gemm, blas::cpu_eigen<float>, 4)->Range(4, 4 << 12);

BENCHMARK_TEMPLATE(BM_batched_gemm, blas::cpu_blas<float>, 32)->Range(4, 4 << 7);
BENCHMARK_TEMPLATE(BM_batched_gemm, blas::cpu_eigen<float>, 32)->Range(4, 4 << 7);

BENCHMARK_MAIN();
