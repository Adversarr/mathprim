#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <iostream>
#include <mathprim/supports/eigen_dense.hpp>

#include "mathprim/core/functional.hpp"

#define MATHPRIM_VERBOSE_MALLOC 1
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;
using namespace mathprim::literal;

__global__ void set_value(float *ptr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    ptr[idx] = static_cast<float>(idx);
    printf("ptr[%d] = %f\n", idx, static_cast<float>(idx));
  }
}

__global__ void get_value(batched<cuda_vec4f32_const_view_t> view) {
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

  // Allocate a pitch memory
  float *ptr = nullptr;
  size_t pitch = 0;
  // cudaMallocPitch use [weight, height] as parameter
  size_t width = 4 * sizeof(float), height = 10 * sizeof(float);
  cudaMallocPitch(&ptr, &pitch, width, height);
  auto pitched_ptr_cuda = make_cudaPitchedPtr(ptr, pitch, width, height);
  std::cout << "pitched_ptr_cuda.ptr = " << pitched_ptr_cuda.ptr
            << ", pitched_ptr_cuda.pitch = " << pitched_ptr_cuda.pitch
            << ", pitched_ptr_cuda.xsize = " << pitched_ptr_cuda.xsize
            << ", pitched_ptr_cuda.ysize = " << pitched_ptr_cuda.ysize
            << std::endl;
  // create view.
  auto view_pitched = from_cuda_pitched_ptr<float>(pitched_ptr_cuda);
  std::cout << "view_pitched=" << view_pitched << std::endl;

  // view back.
  auto pitched_ptr_cuda_back = to_cuda_pitched_ptr(view_pitched);
  std::cout << "pitched_ptr_cuda_back.ptr = " << pitched_ptr_cuda_back.ptr
            << ", pitched_ptr_cuda_back.pitch = " << pitched_ptr_cuda_back.pitch
            << ", pitched_ptr_cuda_back.xsize = " << pitched_ptr_cuda_back.xsize
            << ", pitched_ptr_cuda_back.ysize = " << pitched_ptr_cuda_back.ysize
            << std::endl;

  // Free the memory
  cudaFree(ptr);

  // Make a pitched buffer
  auto pitched_buf = make_cuda_pitched_buffer<float>(make_shape(10, 4));
  std::cout << "pitched_buf=" << pitched_buf.view() << std::endl;

  cudaDeviceSynchronize();

  // cuda streams.
  cudaStream_t stream; cudaStreamCreate(&stream);

  par::cuda parfor_stream(stream);
  par::cuda parfor_default;
  parfor_stream.run(view.shape(), [view] __device__(index_array<2> idx) {
    auto [i, j] = idx;
    printf("Lambda streamd view[%d, %d] = %f\n", i, j, view(i, j));
  });
  parfor_default.run(view.shape(), [view] __device__(index_array<2> idx) {
    auto [i, j] = idx;
    printf("Lambda default view[%d, %d] = %f\n", i, j, view(i, j));
  });

  parfor_stream.sync();
  parfor_default.sync();
  cudaStreamDestroy(stream);

  // vmap:
  auto buf2 = make_buffer<float, device::cuda>(10);

  parfor_default.vmap(
      [] __device__(auto vec4, auto &out) {
        out = vec4[0] + vec4[1] + vec4[2] + vec4[3];
      },
      buf.view(), buf2.view());

  parfor_default.run(buf2.shape(), [view = buf2.view()] __device__(index_t i) {
    // Should be 6, 22, 38, 54, 70, 86, 102, 118, 134, 150 = 16 i + 6
    if (view[i] != 16 * i + 6) {
      printf("Error: buf2[%d] = %f\n", i, view(i));
    } else {
      printf("Ok: buf2[%d] = %f\n", i, view(i));
    }
  });
  parfor_default.sync();

  auto buf3 = make_buffer<float, device::cuda>(10, 3_s);
  buf3.fill_bytes(0);
  parfor_default.run(10, [view = buf3.view()] __device__(index_t i) {
    view(i, 0) = i;
    view(i, 1) = i + 1;
    view(i, 2) = i + 2;
  });
  parfor_default.sync();

  Eigen::Matrix<float, 3, 3> mat;
  mat << 1, 0, 0,  //
  0, 2, 0,     //
  0, 0, 3;
  auto aft = functional::affine_transform<float, device::cuda, 3>(eigen_support::view(mat).as_const());
  aft.bias_[0] = 3;
  aft.bias_[1] = 2;
  aft.bias_[2] = 1;
  parfor_default.vmap(aft, buf3.view());
  parfor_default.run(10, [view = buf3.view()] __device__(index_t i) {
    // should be:
    // i * 1 + 3
    // (i+1) * 2 + 2
    // (i+2) * 3 + 1
    if (view(i, 0) != i + 3 || view(i, 1) != (i + 1) * 2 + 2 || view(i, 2) != (i + 2) * 3 + 1) {
      printf("Error: buf3[%d] = (%f, %f, %f)\n", i, view(i, 0), view(i, 1), view(i, 2));
    } else {
      printf("Ok: buf3[%d] = (%f, %f, %f)\n", i, view(i, 0), view(i, 1), view(i, 2));
    }
  });

  // for (int i = 10; i < 20; ++i) {
  //   index_t total = 1 << i;
  //   auto buf = make_buffer<float, device::cuda>(total);
  //   auto view = buf.view();
  //   parfor_default.run(view.shape(), [view, total] __device__(index_t i) {
  //     auto block_idx = blockIdx.x;
  //     auto thread_idx = threadIdx.x;
  //     auto idx = block_idx * blockDim.x + thread_idx;
  //     if (thread_idx > 0) {
  //       printf("total = %d, block_idx = %d, thread_idx = %d, idx = %d\n", total, block_idx, thread_idx, idx);
  //     }
  //   });
  // }

  return EXIT_SUCCESS;
}
