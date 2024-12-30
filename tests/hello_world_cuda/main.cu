#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>
#include <thread>

#define MATHPRIM_VERBOSE_MALLOC 1
#include <mathprim/core/backends/cuda.cuh>
#include <mathprim/core/common.hpp>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;

void test_convertible(const_f32_buffer_view<3, device_t::cuda> bv) {
  std::cout << "Buffer view: " << bv << std::endl;
}

__global__ void set_value(f32_buffer_view<3, device_t::cuda> bv) {
  int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z;
  auto [X, Y, Z] = bv.shape();
  if (x < X && y < Y && z < Z) {
    bv({x, y, z}) = x * Y * Z + y * Z + z;
  }
}

__global__ void print_value(const_f32_buffer_view<3, device_t::cuda> bv) {
  int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z;
  auto [X, Y, Z] = bv.shape();
  if (x < X && y < Y && z < Z) {
    printf("Value at (%d, %d, %d): %f\n", x, y, z, bv(x, y, z));
  }
}

int main() {
  auto buffer = make_buffer<float, device_t::cuda>({2, 3, 4});
  std::cout << "Buffer: " << buffer << std::endl;
  test_convertible(buffer.view());

  cudaDeviceSynchronize();
  dim3 grid_dim = {2, 3, 4};
  dim3 block_dim = {1, 1, 1};
  set_value<<<grid_dim, block_dim>>>(buffer.view());
  print_value<<<grid_dim, block_dim>>>(buffer.view());

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
  return 0;
}
