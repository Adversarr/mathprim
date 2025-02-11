#include <benchmark/benchmark.h>
#include <iostream>

#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/blas/cublas.cuh"
#include "mathprim/supports/stringify.hpp"
using namespace mathprim;
constexpr index_t dsize = 981; // do not align to 1024.

void cublas_no_pitch_gemm(benchmark::State & state) {
  auto blas = blas::cublas<float>();
  auto mat_a = make_cuda_buffer<float>(dsize, dsize);
  auto mat_b = make_cuda_buffer<float>(dsize, dsize);
  auto mat_c = make_cuda_buffer<float>(dsize, dsize);
  float alpha = 1.0, beta = 0.0;
  auto a = mat_a.view(), b = mat_b.view(), c = mat_c.view();
  std::cout << "a: " << a << std::endl;
  blas.gemm(alpha, a.as_const(), b.as_const(), beta, c);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    blas.gemm(alpha, a.as_const(), b.as_const(), beta, c);
    cudaDeviceSynchronize();
  }
}

void cublas_pitched_gemm(benchmark::State & state) {
  auto blas = blas::cublas<float>();
  auto mat_a = make_cuda_pitched_buffer<float>(make_shape(dsize, dsize));
  auto mat_b = make_cuda_pitched_buffer<float>(make_shape(dsize, dsize));
  auto mat_c = make_cuda_pitched_buffer<float>(make_shape(dsize, dsize));
  float alpha = 1.0, beta = 0.0;
  auto a = mat_a.view(), b = mat_b.view(), c = mat_c.view();
  std::cout << "a: " << a << std::endl;
  blas.gemm(alpha, a.as_const(), b.as_const(), beta, c);
  cudaDeviceSynchronize();

  for (auto _ : state) {
    blas.gemm(alpha, a.as_const(), b.as_const(), beta, c);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(cublas_no_pitch_gemm);
BENCHMARK(cublas_pitched_gemm);

BENCHMARK_MAIN();