
#include <benchmark/benchmark.h>

#include "mathprim/core/backends/cuda.cuh"
#include "mathprim/core/parallel/cuda.cuh"

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/buffer_view.hpp"
#include "mathprim/core/parallel.hpp"

using namespace mathprim;

template <typename Flt> __global__ void par_for(Flt *x, Flt *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * x[i];
  }
}

template <typename Flt> static void par_for_norm(benchmark::State &state) {
  int n = state.range(0);
  int blocksize = 256;
  int gridsize = (n + blocksize - 1) / blocksize;
  Flt *x, *y;
  cudaMalloc(&x, n * sizeof(Flt));
  cudaMalloc(&y, n * sizeof(Flt));

  for (auto _ : state) {
    par_for<<<gridsize, blocksize>>>(x, y, n);
    cudaDeviceSynchronize();
  }
}

template <typename Flt>
static void par_for_norm_mathprim(benchmark::State &state) {
  int n = state.range();

  int blocksize = 256;
  int gridsize = (n + blocksize - 1) / blocksize;

  auto buf_x = make_buffer<Flt, device_t::cuda>(n);
  auto buf_y = make_buffer<Flt, device_t::cuda>(n);

  for (auto _ : state) {
    parfor<par::cuda>::run(
        dim<1>(gridsize), dim<1>(blocksize),
        [x = buf_x.view(), y = buf_y.view(),
         blocksize] __device__(dim<1> block_idx, dim<1> thread_idx) {
          auto this_idx = merge<1>(block_idx, thread_idx, make_dim(blocksize));
          if (is_in_bound(x.shape(), this_idx))
            y(this_idx) = x(this_idx) * x(this_idx);
        });
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(par_for_norm, float)
    ->RangeMultiplier(2)
    ->Range(1 << 20, 1 << 24);
BENCHMARK_TEMPLATE(par_for_norm, double)
    ->RangeMultiplier(2)
    ->Range(1 << 20, 1 << 24);

BENCHMARK_TEMPLATE(par_for_norm_mathprim, float)
    ->RangeMultiplier(2)
    ->Range(1 << 20, 1 << 24);

BENCHMARK_TEMPLATE(par_for_norm_mathprim, double)
    ->RangeMultiplier(2)
    ->Range(1 << 20, 1 << 24);

BENCHMARK_MAIN();
