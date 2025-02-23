#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/stringify.hpp>
namespace mp = mathprim;
using namespace mp::literal;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += temp[lindex + offset];

  // Store the result
  out[gindex] = result;
}

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

int main() {
  auto par_cuda = mp::par::cuda();
  auto host_in = mp::make_buffer<int>(N + 2 * RADIUS),
       host_out = mp::make_buffer<int>(N + 2 * RADIUS);
  auto device_in = mp::make_buffer<int, mp::device::cuda>(N + 2 * RADIUS),
       device_out = mp::make_buffer<int, mp::device::cuda>(N + 2 * RADIUS);

  auto hIn = host_in.view(), hOut = host_out.view();
  auto dIn = device_in.view(), dOut = device_out.view();
  std::fill_n(hIn.data(), N + 2 * RADIUS, 1);
  mp::copy(dIn, hIn);
  mp::copy(dOut, hIn);

  cts_begin(naive, 10);
  stencil_1d<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(dIn.data() + RADIUS,
                                             dOut.data() + RADIUS);
  cts_end(naive);

  cts_begin(mathprim, 10);
  auto grid_size = mathprim::make_shape(N / BLOCK_SIZE);
  auto block_size = mathprim::make_shape(BLOCK_SIZE);
  using index_t = mp::index_t;
  par_cuda.run(grid_size, block_size, [dIn, dOut]__device__(const index_t & block_id, const index_t& thread_id) {
    // auto [block_id] = block_idx;
    // auto [thread_id] = thread_idx;
    mp::index_t i = thread_id + block_id * BLOCK_SIZE;
    __shared__ int shm[BLOCK_SIZE + 2 * RADIUS];
    int lindex = thread_id + RADIUS;
    shm[lindex] = dIn[i + RADIUS];
    if (thread_id < RADIUS) {
      shm[lindex - RADIUS] = dIn[i];
      shm[lindex + BLOCK_SIZE] = dIn[i + BLOCK_SIZE + RADIUS];
    }
    
    __syncthreads();
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += shm[lindex + offset];
    dOut[i + RADIUS] = result;
  });

  cts_end(mathprim);

  mp::copy(hOut, dOut);
  for (auto [i] : hOut.shape()) {
    if (i < RADIUS || i >= N + RADIUS) {
      if (hOut[i] != 1)
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, hOut[i], 1);
    } else {
      if (hOut[i] != 1 + 2 * RADIUS)
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, hOut[i],
               1 + 2 * RADIUS);
    }
  }
  return 0;
}