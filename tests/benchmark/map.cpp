#include "mathprim/core/buffer.hpp"
#include <benchmark/benchmark.h>

#include <mathprim/supports/eigen_dense.hpp>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;
static void BM_Eigen_VectorMap(benchmark::State &state) {
  auto buffer = make_buffer<float>(shape_t<keep_dim, 4>(state.range(0), 4)); // nx4
//   std::cout << buffer.view().shape() << std::endl;
  auto view = buffer.view();
  for (auto [i, j] : view.shape()) {
    view(i, j) = i * 4 + j;
  }

  for (auto _ : state) {
    for (auto v : view) {
      auto map = eigen_support::cmap(v.as_const());
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
      Eigen::Vector4f v2 = map * 2;
      benchmark::DoNotOptimize(v2);
    }
  }
}

static void BM_Eigen_VectorMapNaive(benchmark::State &state) {
  auto buffer = make_buffer<float>(shape_t<keep_dim, 4>(state.range(0), 4)); // nx4
  auto view = buffer.view();
  for (auto [i, j] : view.shape()) {
    view(i, j) = i * 4 + j;
  }
  for (auto _ : state) {
    for (index_t i = 0; i < view.shape()[0]; ++i) {
      auto map = Eigen::Map<Eigen::Vector4f>(view.data() + i * 4);
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
      Eigen::Vector4f v = map * 2;
      benchmark::DoNotOptimize(v);
    }
  }
}

static void BM_Eigen_MatrixMap(benchmark::State &state) {
  auto buffer = make_buffer<float>(shape_t<keep_dim, 4, 4>(state.range(0), 4, 4)); // nx4x4
//   std::cout << buffer.view().shape() << std::endl;
  auto view = buffer.view();
  for (auto [i, j, k] : view.shape()) {
    view(i, j, k) = i * 16 + j * 4 + k;
  }
  for (auto _ : state) {
    for (auto v : view.as_const()) {
      auto map = eigen_support::cmap(v);
      float norm = map.norm();
      benchmark::DoNotOptimize(norm);
    }
  }
}

static void BM_Eigen_MatrixMapNaive(benchmark::State &state) {
  auto buffer = make_buffer<float>(shape_t<keep_dim, 16>(state.range(0), 16)); // nx16
  auto view = buffer.view();
  for (auto [i, j] : view.shape()) {
    view(i, j) = i * 16 + j;
  }
  for (auto _ : state) {
    for (index_t i = 0; i < view.shape()[0]; ++i) {
      auto map = Eigen::Map<const Eigen::Matrix4f>(view.data() + i * 16);
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
