#include "../cts_helper.cuh"
#include <thrust/reduce.h>

const size_t N = 8ULL * 1024ULL * 1024ULL; // data size
// const size_t N = 256*640; // data size
const int BLOCK_SIZE = 256; // CUDA maximum is 1024

__global__ void atomic_add(float *a, const float *b, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(a, b[i]);
  }
}

__global__ void reduce_a(float *gdata, float *out) {
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  sdata[tid] = 0.0f;
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  while (idx < N) { // grid stride loop to load data
    sdata[tid] += gdata[idx];
    idx += gridDim.x * blockDim.x;
  }

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) // parallel sweep reduction
      sdata[tid] += sdata[tid + s];
  }
  if (tid == 0)
    atomicAdd(out, sdata[0]);
}

int main() {
  auto buf_a = mp::make_buffer<float, mp::device::cuda>(N);
  auto buf_out = mp::make_buffer<float, mp::device::cuda>(1);
  auto a = buf_a.view();
  auto out = buf_out.view();
  auto block_size = mp::make_shape(BLOCK_SIZE);
  auto grid_size = mp::make_shape(N / BLOCK_SIZE);

  mp::par::cuda().run(a.shape(),
                      [a] __device__(const auto &idx) { a(idx) = 1.0f; });

  cts_begin(atomic_add, 10);
  atomic_add<<<grid_size[0], block_size[0]>>>(out.data(), a.data(), N);
  cts_end(atomic_add); // 15~16ms

  cts_begin(thrust, 10);
  float sum = thrust::reduce(thrust::device, a.data(), a.data() + N, 0.0f,
                             thrust::plus<float>());
  cts_end(thrust); // 0.12~0.13ms

  cts_begin(cublas, 10);
  auto blas = mp::blas::cublas<float>();
  blas.asum(a.as_const());
  cts_end(cublas); // 0.11~0.12ms

  cts_begin(reduce_a, 10);
  reduce_a<<<grid_size[0], block_size[0]>>>(a.data(), out.data());
  cts_end(reduce_a); // 0.33~0.35ms

  return EXIT_SUCCESS;
}