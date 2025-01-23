#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>

#define MATHPRIM_VERBOSE_MALLOC 1
#include <mathprim/core/backends/cuda.cuh>
#include <mathprim/core/common.hpp>
#include <mathprim/core/parallel.hpp>
#include <mathprim/core/parallel/cuda.cuh>
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

__global__ void check_parallel_assignment(const_f32_buffer_view<4, device_t::cuda> bv) {
  int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z;
  auto [X, Y, Z, W] = bv.shape();
  if (x < X && y < Y && z < Z) {
    assert(bv(x, y, z, 0) == x * Y * Z + y * Z + z);
  }
}

template <typename Fn> __global__ void lambda_call(Fn fn) {
  fn();
}

void lambda_call_lambda() {
  auto lambda = [] __device__() {
    printf("Hello from lambda\n");
  };
  lambda_call<<<1, 1>>>([l = lambda] __device__() {
    l();
  });
}

int main() {
  auto buffer = make_buffer<float, 3, device_t::cuda>(dim{2, 3, 4});
  std::cout << "Buffer: " << buffer << std::endl;
  test_convertible(buffer.view());

  cudaDeviceSynchronize();
  dim3 grid_dim = {2, 3, 4};
  dim3 block_dim = {1, 1, 1};
  set_value<<<grid_dim, block_dim>>>(buffer.view());
  print_value<<<grid_dim, block_dim>>>(buffer.view());

  auto buffer2 = make_buffer<float, 4, device_t::cuda>(dim_t{4, 3, 2, 1});

  parfor_cuda::run(dim_t(4, 3, 2, 1), dim_t(1), [bv = buffer2.view()] __device__(dim_t grid_id, dim_t /* block_id */) {
    bv(grid_id) = grid_id.x_ * 3 * 2 * 1 + grid_id.y_ * 2 * 1 + grid_id.z_ * 1 + grid_id.w_;
  });

  parfor_cuda::for_each_indexed(buffer2.view(), [] __device__(auto idx, auto value) {
    assert(value == idx.x_ * 3 * 2 * 1 + idx.y_ * 2 * 1 + idx.z_ * 1 + idx.w_);
  });

  parfor_cuda::vmap(
      [] __device__(auto v1, auto v2) {
        printf("v1.shape=(%d, %d)\n", v1.shape(0), v1.shape(1));
        printf("v2.shape=(%d, %d, %d)\n", v2.shape(0), v2.shape(1), v2.shape(2));
      },
      make_vmap_arg(buffer.view()), make_vmap_arg<2>(buffer2.view()));

  check_parallel_assignment<<<dim3(4, 3, 2), dim3(1)>>>(buffer2.view());

  lambda_call_lambda();

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
  return 0;
}
