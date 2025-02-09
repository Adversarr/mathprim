// vector_add.cu
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>

#include "../cts_helper.cuh"

namespace mp = mathprim;
using namespace mp::literal;

constexpr mp::index_t DSIZE = 32 * 1048576;

__global__ void vector_add_naive(float *a, float *b, float *c,
                                 mp::index_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_strided(float *a, float *b, float *c,
                                   mp::index_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < size; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  auto par_cuda = mp::par::cuda();
  auto par_host = mp::par::seq();
  auto host_A = mp::make_buffer<float>(DSIZE);
  auto host_B = mp::make_buffer<float>(DSIZE);
  auto host_C = mp::make_buffer<float>(DSIZE);
  auto device_A = mp::make_buffer<float, mp::device::cuda>(DSIZE);
  auto device_B = mp::make_buffer<float, mp::device::cuda>(DSIZE);
  auto device_C = mp::make_buffer<float, mp::device::cuda>(DSIZE);

  auto hA = host_A.view(), hB = host_B.view(), hC = host_C.view();
  auto dA = device_A.view(), dB = device_B.view(), dC = device_C.view();

  auto total = mp::make_shape(DSIZE);
  par_host.run(total, [a = hA, b = hB](auto idx) {
    auto [i] = idx;
    a[i] = b[i] = static_cast<float>(i);
  });
  mp::copy(dA, hA);
  mp::copy(dB, hB);
  mp::copy(dC, hC);

  auto blocks = mp::make_shape(128);
  auto grids = mp::up_div(total, blocks);

  for (auto blk : {8, 16, 32, 64, 128, 256, 512, 1024}) {
    for (auto grd : {8, 16, 32, 64, 128, 256, 512, 1024}) {
      blocks = mp::make_shape(blk);
      grids = mp::make_shape(grd);
      std::cout << "Blocks: " << blocks << ", Grids: " << grids << std::endl;
      cts_begin(mathprim, 10);
      par_cuda.run(
          grids, blocks,
          [a = dA, b = dB, c = dC, blocks,
           grids] __device__(const auto &block_idx, const auto &thread_idx) {
            const auto start = block_idx[0] * blocks[0] + thread_idx[0];
            for (auto i = start; i < DSIZE; i += grids[0] * blocks[0]) {
              c[i] = a[i] + b[i];
            }
          });
      cts_end(mathprim);
    }
  }

  cts_begin(cublas, 10);
  mp::blas::cublas<float> cublas;
  cublas.axpy(1.0, dA.as_const(), dB);
  cts_end(cublas);
  // copy back to host
  mp::copy(hC, dC);
  par_host.run(mp::make_shape(DSIZE), [hA, hB, hC](auto idx) {
    auto [i] = idx;
    if (hC[i] != hA[i] + hB[i]) {
      printf("Error: %f + %f != %f\n", hA[i], hB[i], hC[i]);
      abort();
    }
  });

  MATHPRIM_CUDA_CHECK_SUCCESS(cudaDeviceSynchronize());
  return EXIT_SUCCESS;
}
