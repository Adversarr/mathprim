#include <benchmark/benchmark.h>

#include <mathprim/supports/eigen_dense.hpp>

using namespace mathprim;
static void BM_Eigen_VectorMap(benchmark::State& state) {
  auto buffer
      = make_buffer<float>(static_cast<index_t>(state.range(0)), 3);  // nx3
  auto view = buffer.view();
  for (auto [i, j] : view.shape()) {
    view(i, j) = i * 3 + j;
  }
  for (auto _ : state) {
    for (auto v : view) {
      auto map = eigen_support::cmap<3>(v.as_const());
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
    }
  }
}

static void BM_Eigen_VectorMapNaive(benchmark::State& state) {
  auto buffer
      = make_buffer<float>(static_cast<index_t>(state.range(0)), 3);  // nx3
  auto view = buffer.view();
  for (auto [i, j] : view.shape()) {
    view(i, j) = i * 3 + j;
  }
  for (auto _ : state) {
    for (index_t i = 0; i < view.shape()[0]; ++i) {
      auto map = Eigen::Map<Eigen::Vector3f>(view.data() + i * 3);
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
    }
  }
}

static void BM_Eigen_MatrixMap(benchmark::State& state) {
  auto buffer
      = make_buffer<float>(static_cast<index_t>(state.range(0)), 4, 4);  // nx16
  auto view = buffer.view();
  for (auto [i, j, k] : view.shape()) {
    view(i, j, k) = i * 16 + j * 4 + k;
  }
  for (auto _ : state) {
    for (auto v : view) {
      auto map = eigen_support::cmap<4, 4>(v.as_const());
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
    }
  }
}

static void BM_Eigen_MatrixMapNaive(benchmark::State& state) {
  auto buffer
      = make_buffer<float>(static_cast<index_t>(state.range(0)), 16);  // nx16
  auto view = buffer.view();
  for (auto [i, j] : view.shape()) {
    view(i, j) = i * 16 + j;
  }
  for (auto _ : state) {
    for (index_t i = 0; i < view.shape()[0]; ++i) {
      auto map = Eigen::Map<Eigen::Matrix<float, 4, 4>>(view.data() + i * 16);
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
    }
  }
}

BENCHMARK(BM_Eigen_MatrixMap)->Range(8, 8 << 16);
BENCHMARK(BM_Eigen_MatrixMapNaive)->Range(8, 8 << 16);


BENCHMARK(BM_Eigen_VectorMap)->Range(8, 8 << 16);
BENCHMARK(BM_Eigen_VectorMapNaive)->Range(8, 8 << 16);

BENCHMARK_MAIN();