#include <benchmark/benchmark.h>

#include <cmath>
#include <mathprim/core/blas/cpu_blas.hpp>
#include <mathprim/core/common.hpp>
#include <mathprim/core/parallel.hpp>
#include <mathprim/core/parallel/openmp.hpp>

using namespace mathprim;

template <typename Flt, par p>
static void BM_par_for_norm(benchmark::State& state) {
  index_t n = static_cast<index_t>(state.range(0));
  auto buffer = make_buffer<Flt>(n, 3);

  for (auto _ : state) {
    mathprim::parfor<p>::run(dim<1>{n}, [&](dim<1> ii) {
      float norm = 0;
      index_t i = ii[0];
      auto buf_i = buffer.data() + i * 3;
      norm = buf_i[0] * buf_i[0] + buf_i[1] * buf_i[1] + buf_i[2] * buf_i[2];
      norm = std::sqrt(norm);
      benchmark::DoNotOptimize(norm);
    });
  }
}

template <typename Flt>
static void BM_par_for_norm_omp(benchmark::State& state) {
  index_t n = static_cast<index_t>(state.range(0));
  auto buffer = make_buffer<Flt>(n, 3);

  for (auto _ : state) {
    auto view = buffer.view();
    index_t chunk_size = n / omp_get_max_threads();
    #pragma omp parallel for schedule(static, chunk_size)
    for (index_t i = 0; i < n; ++i) {
      float norm = 0;
      auto buf_i = view.data() + i * 3;
      norm = buf_i[0] * buf_i[0] + buf_i[1] * buf_i[1] + buf_i[2] * buf_i[2];
      norm = std::sqrt(norm);
      benchmark::DoNotOptimize(norm);
    }
  }
}

constexpr index_t min_size = 1 << 5;
constexpr index_t max_size = 1 << 20;

BENCHMARK_TEMPLATE(BM_par_for_norm, float, par::seq)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK(BM_par_for_norm_omp<float>)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);

BENCHMARK_TEMPLATE(BM_par_for_norm, float, par::openmp)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, float, par::std)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, double, par::seq)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, double, par::openmp)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, double, par::std)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);

template <typename Flt, par p>
static void BM_par_axpy(benchmark::State& state) {
  index_t n = static_cast<index_t>(state.range(0));
  auto buffer_x = make_buffer<Flt>(n);
  auto buffer_y = make_buffer<Flt>(n);
  Flt a = 2.0;
  for (auto _ : state) {
    mathprim::parfor<p>::run(dim_t{n}, [x = buffer_x.view(), y = buffer_y.view(), a](dim_t ii) {
      index_t i = ii[0];
      y[i] += a * x[i];
    });
  }
}

template <typename Flt> static void BM_axpy_blas_blas(benchmark::State& state) {
  index_t n = static_cast<index_t>(state.range(0));
  auto buffer_x = make_buffer<Flt>(n);
  auto buffer_y = make_buffer<Flt>(n);
  float a = 2.0;
  for (auto _ : state) {
    using blas_ = blas::blas_impl_cpu_blas<Flt>;
    blas_::axpy(a, buffer_x.view(), buffer_y.view());
  }
}

BENCHMARK_TEMPLATE(BM_par_axpy, float, par::seq)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_axpy, float, par::openmp)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_axpy, float, par::std)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_axpy, double, par::seq)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_axpy, double, par::openmp)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_axpy, double, par::std)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_axpy_blas_blas, float)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_axpy_blas_blas, double)
    ->RangeMultiplier(2)
    ->Range(min_size, max_size);

BENCHMARK_MAIN();
