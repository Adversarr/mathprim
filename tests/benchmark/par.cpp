#include <benchmark/benchmark.h>

#include <cmath>
#include <mathprim/core/common.hpp>
#include <mathprim/core/parallel.hpp>
#include <mathprim/core/parallel/openmp.hpp>

using namespace mathprim;

template <typename Flt, par p>
static void BM_par_for_norm(benchmark::State& state) {
  index_t n = static_cast<index_t>(state.range(0));
  auto buffer = make_buffer<Flt>(n, 3);

  for (auto _ : state) {
    mathprim::parfor<p>::run(dim_t{n}, [&](dim_t ii) {
      float norm = 0;
      index_t i = ii[0];
      auto buf_i = buffer.data() + i * 3;
      norm = buf_i[0] * buf_i[0] + buf_i[1] * buf_i[1] + buf_i[2] * buf_i[2];
      norm = std::sqrt(norm);
      benchmark::DoNotOptimize(norm);
    });
  }
}

constexpr index_t min_size = 1 << 5;
constexpr index_t max_size = 1 << 20;

BENCHMARK_TEMPLATE(BM_par_for_norm, float, par::seq)->RangeMultiplier(2)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, float, par::openmp)->RangeMultiplier(2)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, float, par::std)->RangeMultiplier(2)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, double, par::seq)->RangeMultiplier(2)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, double, par::openmp)->RangeMultiplier(2)->Range(min_size, max_size);
BENCHMARK_TEMPLATE(BM_par_for_norm, double, par::std)->RangeMultiplier(2)->Range(min_size, max_size);

BENCHMARK_MAIN();
