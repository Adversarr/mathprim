#include <mathprim/parallel/cuda.cuh>
namespace mp = mathprim;
using namespace mp::literal;
int main() {
  auto par = mp::par::cuda();

  auto dimensions = mp::make_shape(2);
  par.run(dimensions, dimensions, [] __device__ (auto block_idx, auto thread_idx) {
    printf("Bidx, Tidx: %d, %d == %d, %d\n", block_idx[0], thread_idx[0], blockIdx.x, threadIdx.x);
  });

  MATHPRIM_CUDA_CHECK_SUCCESS(cudaDeviceSynchronize());
  return EXIT_SUCCESS;
}