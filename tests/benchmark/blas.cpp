#include <benchmark/benchmark.h>

#include <mathprim/core/blas.hpp>
#include <mathprim/core/blas/cpu_blas.hpp>
#include <mathprim/core/blas/cpu_handmade.hpp>
#include <mathprim/supports/eigen_dense.hpp>

#include "mathprim/core/blas/cpu_eigen.hpp"
#include "mathprim/core/buffer.hpp"

using namespace mathprim;

static void BM_axpy_handmade(benchmark::State& state) {
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  memset(x, 0);
  memset(y, 0);

  for (auto _ : state) {
    blas::blas_impl_cpu_handmade<float>::axpy(1.0f, x.view().as_const(),
                                              y.view());
  }
}

static void BM_axpy_eigen(benchmark::State& state) {
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  memset(x, 0);
  memset(y, 0);

  for (auto _ : state) {
    blas::blas_impl_cpu_eigen<float>::axpy(1.0f, x.view(), y.view());
  }
}

static void BM_axpy_blas(benchmark::State& state) {
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  memset(x, 0);
  memset(y, 0);

  for (auto _ : state) {
    blas::blas_impl_cpu_blas<float>::axpy(1.0f, x.view().as_const(), y.view());
  }
}

static void BM_gemv_handmade(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  memset(A, 0);
  memset(x, 0);
  memset(y, 0);
  auto a_view = A.view().as_const();
  auto x_view = x.view().as_const();
  auto y_view = y.view();
  for (auto _ : state) {
    blas::blas_impl_cpu_handmade<float>::gemv(1.0f, a_view, x_view, 1.0f,
                                              y_view);
  }
}

static void BM_gemm_handmade(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto B = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto C = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  memset(A, 0);
  memset(B, 0);
  memset(C, 0);
  auto a_view = A.view().as_const();
  auto b_view = B.view().as_const();
  auto c_view = C.view();

  for (auto _ : state) {
    blas::blas_impl_cpu_handmade<float>::gemm(1.0f, a_view, b_view, 1.0f,
                                              c_view);
  }
}

static void BM_gemv_eigen(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  memset(A, 0);
  memset(x, 0);
  memset(y, 0);
  for (auto _ : state) {
    blas::blas_impl_cpu_eigen<float>::gemv(1.0f, A.view(), x.view(), 1.0f,
                                           y.view());
  }
}

static void BM_gemm_eigen(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto B = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto C = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  memset(A, 0);
  memset(B, 0);
  memset(C, 0);
  auto a_view = A.view().as_const();
  auto b_view = B.view().as_const();
  auto c_view = C.view();
  for (auto _ : state) {
    blas::blas_impl_cpu_eigen<float>::gemm(1.0f, a_view, b_view, 1.0f, c_view);
  }
}

static void BM_gemv_blas(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto x = make_buffer<float>(static_cast<index_t>(state.range(0)));
  auto y = make_buffer<float>(static_cast<index_t>(state.range(0)));
  memset(A, 0);
  memset(x, 0);
  memset(y, 0);
  auto a_view = A.view().as_const();
  auto x_view = x.view().as_const();
  auto y_view = y.view();

  for (auto _ : state) {
    blas::blas_impl_cpu_blas<float>::gemv(1.0f, a_view, x_view, 1.0f, y_view);
  }
}

static void BM_gemm_blas(benchmark::State& state) {
  auto A = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto B = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  auto C = make_buffer<float>(static_cast<index_t>(state.range(0)),
                              static_cast<index_t>(state.range(0)));
  memset(A, 0);
  memset(B, 0);
  memset(C, 0);
  auto a_view = A.view().as_const();
  auto b_view = B.view().as_const();
  auto c_view = C.view();

  for (auto _ : state) {
    blas::blas_impl_cpu_blas<float>::gemm(1.0f, a_view, b_view, 1.0f, c_view);
  }
}

BENCHMARK(BM_gemv_handmade)->Range(8, 8 << 10);
BENCHMARK(BM_gemv_eigen)->Range(8, 8 << 10);
BENCHMARK(BM_gemv_blas)->Range(8, 8 << 10);
BENCHMARK(BM_gemm_handmade)->Range(8, 8 << 7);
BENCHMARK(BM_gemm_eigen)->Range(8, 8 << 7);
BENCHMARK(BM_gemm_blas)->Range(8, 8 << 7);

BENCHMARK(BM_axpy_handmade)->Range(8, 8 << 14);
BENCHMARK(BM_axpy_eigen)->Range(8, 8 << 14);
BENCHMARK(BM_axpy_blas)->Range(8, 8 << 14);
BENCHMARK_MAIN();
