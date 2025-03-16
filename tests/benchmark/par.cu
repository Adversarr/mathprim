
#include <benchmark/benchmark.h>

#include "mathprim/parallel/cuda.cuh"
#include "mathprim/core/devices/cuda.cuh"

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
void par_for_norm_mathprim(benchmark::State &state) {
  int n = state.range();

  int blocksize = 256;
  int gridsize = (n + blocksize - 1) / blocksize;

  auto buf_x = make_cuda_buffer<Flt>(n);
  auto buf_y = make_cuda_buffer<Flt>(n);

  for (auto _ : state) {
    par::cuda().run(
        make_shape(gridsize), make_shape(blocksize),
        [x = buf_x.view(), y = buf_y.view(),
         blocksize] __device__(auto block_idx, auto thread_idx) {
          auto this_idx = block_idx[0] * blocksize + thread_idx[0];
          if (is_in_bound(x.shape(), index_array<1>{this_idx}))
            y(this_idx) = x(this_idx) * x(this_idx);
        });
    cudaDeviceSynchronize();
  }
}
template <typename Scalar>
struct sqr_out {
  MATHPRIM_PRIMFUNC Scalar operator()(Scalar x) const { return x * x; }
};

template <typename Flt>
void par_for_norm_vmap(benchmark::State &state) {
  int n = state.range();

  int blocksize = 256;
  int gridsize = (n + blocksize - 1) / blocksize;

  auto buf_x = make_cuda_buffer<Flt>(n);
  auto buf_y = make_cuda_buffer<Flt>(n);

  auto pf = par::cuda();

  for (auto _ : state) {
    pf.vmap(par::make_output_vmapped(sqr_out<Flt>()), buf_y.view(), buf_x.view());
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

BENCHMARK_TEMPLATE(par_for_norm_vmap, float)
    ->RangeMultiplier(2)
    ->Range(1 << 20, 1 << 24);

BENCHMARK_TEMPLATE(par_for_norm_vmap, double)
    ->RangeMultiplier(2)
    ->Range(1 << 20, 1 << 24);


BENCHMARK_MAIN();
