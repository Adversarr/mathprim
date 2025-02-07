#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>

#define MATHPRIM_VERBOSE_MALLOC 1
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
// #include <mathprim/supports/stringify.hpp>

using namespace mathprim;
using namespace mathprim::literal;

__global__ void set_value(float *ptr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    ptr[idx] = static_cast<float>(idx);
    printf("ptr[%d] = %f\n", idx, static_cast<float>(idx));
  }
}

__global__ void get_value(field_t<cuda_vec4f32_const_view_t> view) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    printf("view.size() = %d\n", view.size());
    printf("view.shape() = (%d, %d)\n", view.shape(0), view.shape(1));
  }

  if (idx < view.size()) {
    auto i = idx / 4;
    auto j = idx % 4;
    printf("view[%d][%d] = %f\n", i, j, view(i, j));
  }
}

int main() {
  auto buf = make_buffer<float, device::cuda>(10, 4_s);
  auto view = buf.view();
  std::cout << view.size() << std::endl;
  auto [i, j] = view.shape();
  std::cout << i << " " << j << std::endl;
  set_value<<<view.size(), 1>>>(buf.data(), buf.size());
  get_value<<<view.size(), 1>>>(view);

  par::cuda parfor;

  parfor.run(view.shape(), [view] __device__(index_array<2> idx) {
    auto [i, j] = idx;
    printf("Lambda view[%d, %d] = %f\n", i, j, view(i, j));
  });

  parfor.run(dshape<4>(10, 4, 1, 1), [view] __device__(index_array<4> idx) {
    auto [i, j, k, l] = idx;
    printf("Lambda view[%d, %d, %d, %d] = %f\n", i, j, k, l, view(i, j));
  });

  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}
