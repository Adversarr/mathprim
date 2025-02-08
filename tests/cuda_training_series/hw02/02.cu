#include <stdio.h>

// these are just for timing measurments
#include "../cts_helper.cuh"
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int DSIZE = 8192;
constexpr int block_size = 32; // CUDA maximum is 1024 *total* threads in block
constexpr float A_val = 3.0f;
constexpr float B_val = 2.0f;

#define INDX(i, j) ((i) * ds + (j))

__global__ void mmul_naive(const float *A, const float *B, float *C, int ds) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)) {
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[INDX(idy, i)] * B[INDX(i, idx)];
    C[INDX(idy, idx)] = temp;
  }
}

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul_shared(const float *A, const float *B, float *C, int ds) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)) {
    float temp = 0;
    for (int i = 0; i < ds / block_size; i++) {

      int row_a = idy, col_a = i * block_size + threadIdx.x;
      int col_b = idx, row_b = i * block_size + threadIdx.y;
      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[INDX(row_a, col_a)];
      Bs[threadIdx.y][threadIdx.x] = B[INDX(row_b, col_b)];

      // Synchronize
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
        temp += As[threadIdx.y][k] *
                Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();
    }

    // Write to global memory
    C[INDX(idy, idx)] = temp;
  }
}

int main() {

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;

  // start timing
  t0 = clock();

  h_A = new float[DSIZE * DSIZE];
  h_B = new float[DSIZE * DSIZE];
  h_C = new float[DSIZE * DSIZE];
  for (int i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);
  cts_begin(naive, 10);
  mmul_naive<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cts_end(naive);

  cts_begin(mathprim, 10);
  auto dA = mp::view<mp::device::cuda>(d_A, mp::make_shape(DSIZE, DSIZE));
  auto dB = mp::view<mp::device::cuda>(d_B, mp::make_shape(DSIZE, DSIZE));
  auto dC = mp::view<mp::device::cuda>(d_C, mp::make_shape(DSIZE, DSIZE));
  auto par = mp::par::cuda();
  auto grid_s = mp::make_shape(grid.x, grid.y);
  auto block_s = mp::make_shape(block.x, block.y);
  par.run(
      grid_s, block_s,
      [dA, dB, dC] __device__(const auto &block_idx, const auto &thread_idx) {
        __shared__ float As[block_size][block_size];
        __shared__ float Bs[block_size][block_size];
        auto [th_x, th_y] = thread_idx;
        auto [bl_x, bl_y] = block_idx;
        int idx = th_x + block_size * bl_x;
        int idy = th_y + block_size * bl_y;
        float temp = 0;
        for (int i = 0; i < DSIZE / block_size; i++) {
          int row_a = idy, col_a = i * block_size + th_x;
          int col_b = idx, row_b = i * block_size + th_y;
          As[th_y][th_x] = dA(row_a, col_a);
          Bs[th_y][th_x] = dB(row_b, col_b);
          __syncthreads();
          for (int k = 0; k < block_size; k++)
            temp += As[th_y][k] * Bs[k][th_x];
          __syncthreads();
        }
        dC(idx, idy) = temp;
      });
  cts_end(mathprim);

  cts_begin(shared, 10);
  mmul_shared<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cts_end(shared);

  cts_begin(cublas, 10);
  mp::blas::cublas<float> cublas;
  auto dA = mp::view<mp::device::cuda>(d_A, mp::make_shape(DSIZE, DSIZE));
  auto dB = mp::view<mp::device::cuda>(d_B, mp::make_shape(DSIZE, DSIZE));
  auto dC = mp::view<mp::device::cuda>(d_C, mp::make_shape(DSIZE, DSIZE));
  cublas.gemm(1.0, dA.as_const(), dB.as_const(), 0.0, dC);
  cts_end(cublas);

  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE * DSIZE; i++)
    if (h_C[i] != A_val * B_val * DSIZE) {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i],
             A_val * B_val * DSIZE);
      return -1;
    }
  printf("Success!\n");
  return 0;
}
