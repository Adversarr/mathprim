// matrix_mul.cu
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>
#include <mathprim/blas/cublas.cuh>
namespace mp = mathprim;
using namespace mp::literal;

constexpr mp::index_t DSIZE = 1024;

int main() {
  auto par_cuda = mp::par::cuda();
  auto par_host = mp::par::seq();
  auto host_A = mp::make_buffer<float>(DSIZE, DSIZE);
  auto host_B = mp::make_buffer<float>(DSIZE, DSIZE);
  auto host_C = mp::make_buffer<float>(DSIZE, DSIZE);
  auto device_A = mp::make_buffer<float, mp::device::cuda>(DSIZE, DSIZE);
  auto device_B = mp::make_buffer<float, mp::device::cuda>(DSIZE, DSIZE);
  auto device_C = mp::make_buffer<float, mp::device::cuda>(DSIZE, DSIZE);

  auto hA = host_A.view(), hB = host_B.view(), hC = host_C.view();
  auto dA = device_A.view(), dB = device_B.view(), dC = device_C.view();

  auto total = mp::make_shape(DSIZE, DSIZE);
  par_host.run(total, [a = hA, b = hB](auto idx) {
    auto [i, j] = idx;
    a(i, j) = b(i, j) = static_cast<float>(i + j) / 3000.0f;
  });
  mp::copy(dA, hA);
  mp::copy(dB, hB);
  mp::copy(dC, hC);

  auto blocks = mp::make_shape(16, 16);
  auto grids = mp::up_div(total, blocks);

  cudaEvent_t start, stop;
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&start));
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&stop));


  for (int i = 0; i < 10; i++) {
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(start));
    mp::blas::cublas<float> blas;
    blas.gemm(1.0, dA.as_const(), dB.as_const(), 0.0, dC);
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(stop));
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventSynchronize(stop));
    float milliseconds = 0;
    MATHPRIM_CUDA_CHECK_SUCCESS(
        cudaEventElapsedTime(&milliseconds, start, stop));
    printf("(cublas) Elapsed time: %f ms\n", milliseconds);
  }


  for (int i = 0; i < 10; i++) {
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(start));
    // vector addition
    par_cuda.run(grids, blocks,
                 [a = dA, b = dB, c = dC, blocks] __device__(
                     const auto &block_idx, const auto &thread_idx) {
                   auto [i, j] = block_idx * blocks.to_array() + thread_idx;
                   float sum = 0.0f;
                   for (int k = 0; k < DSIZE; k++) {
                     sum += a(i, k) * b(k, j);
                   }
                   c(i, j) = sum;
                 });
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(stop));
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventSynchronize(stop));
    float milliseconds = 0;
    MATHPRIM_CUDA_CHECK_SUCCESS(
        cudaEventElapsedTime(&milliseconds, start, stop));
    printf("(mathprim) Elapsed time: %f ms\n", milliseconds);
  }

  // copy back to host
  mp::copy(hC, dC);
  par_host.run(total, [hA, hB, hC](auto idx) {
    auto [i, j] = idx;
    float sum = 0.0f;
    for (int k = 0; k < DSIZE; k++) {
      sum += hA(i, k) * hB(k, j);
    }
    if (std::abs(hC(i, j) - sum) > 1e-2) {
      printf("Mismatch at (%d, %d): %f != %f\n", i, j, hC(i, j), sum);
    }
  });

  MATHPRIM_CUDA_CHECK_SUCCESS(cudaDeviceSynchronize());
  return EXIT_SUCCESS;
}