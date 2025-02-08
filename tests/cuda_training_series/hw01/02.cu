// vector_add.cu
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>
namespace mp = mathprim;
using namespace mp::literal;

constexpr mp::index_t DSIZE = 4096;

__global__ void vector_add_naive(
  float* a, float* b, float* c, mp::index_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
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
  std::cout << "Blocks: " << blocks << ", Grids: " << grids << std::endl;
  cudaEvent_t start, stop;
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&start));
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&stop));
  for (int i = 0; i < 10; i++) {
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(start));
    // vector addition
    par_cuda.run(grids, blocks,
                 [a = dA, b = dB, c = dC, blocks] __device__(
                     const auto &block_idx, const auto &thread_idx) {
                   auto i = block_idx[0] * blocks[0] + thread_idx[0];
                   c[i] = a[i] + b[i];
                 });
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(stop));
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventSynchronize(stop));
    float milliseconds = 0;
    MATHPRIM_CUDA_CHECK_SUCCESS(
        cudaEventElapsedTime(&milliseconds, start, stop));
    printf("(mathprim) Elapsed time: %f ms\n", milliseconds);
  }
  for (int i = 0; i < 10; i++) {
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(start));

    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(start));
    vector_add_naive<<<grids[0], blocks[0]>>>(dA.data(), dB.data(), dC.data(),
                                              DSIZE);
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(stop));
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventSynchronize(stop));
    float milliseconds = 0;
    MATHPRIM_CUDA_CHECK_SUCCESS(
        cudaEventElapsedTime(&milliseconds, start, stop));
    printf("(naive) Elapsed time: %f ms\n", milliseconds);
  }

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