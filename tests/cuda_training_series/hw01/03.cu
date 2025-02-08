// matrix_mul.cu
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>
namespace mp = mathprim;
using namespace mp::literal;

constexpr mp::index_t DSIZE = 1024;

#define cts_begin(name, loop_count)                                            \
  cudaEvent_t name##_start, name##_stop;                                       \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&name##_start));                 \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventCreate(&name##_stop));                  \
  for (int i = 0; i < (loop_count); i++) {                                     \
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(name##_start))
#define cts_end(name)                                                          \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventRecord(name##_stop));                   \
  MATHPRIM_CUDA_CHECK_SUCCESS(cudaEventSynchronize(name##_stop));              \
  float milliseconds = 0;                                                      \
  MATHPRIM_CUDA_CHECK_SUCCESS(                                                 \
      cudaEventElapsedTime(&milliseconds, name##_start, name##_stop));         \
  printf("(%s) Elapsed time: %f ms\n", #name, milliseconds);                   \
  }                                                                            \
  do {                                                                         \
  } while (0)

__global__ void matmul_naive(float *a, float *b, float *c, mp::index_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < size && j < size) {
    float sum = 0.0f;
    for (int k = 0; k < size; k++) {
      sum += a[i * size + k] * b[k * size + j];
    }
    c[i * size + j] = sum;
  }
}

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

  cts_begin(cublas, 10);
  auto blas = mp::blas::cublas<float>();
  blas.gemm(1.0, dA.as_const(), dB.as_const(), 0.0, dC);
  cts_end(cublas);

  cts_begin(mathprim, 10);
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
  cts_end(mathprim);

  dim3 cuda_blocks(16, 16);
  dim3 cuda_grids(grids[0], grids[1]);
  cts_begin(cuda, 10);
  matmul_naive<<<cuda_grids, cuda_blocks>>>(dA.data(), dB.data(), dC.data(),
                                            DSIZE);
  cts_end(cuda);

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