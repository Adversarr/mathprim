#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/eigen_dense.hpp>

using namespace mathprim;

int main() {
  auto matrix = make_buffer<float, device::cuda>(4, 4);
  auto host_matrix = make_buffer<float>(4, 4);
  auto m = matrix.view();
  auto hm = host_matrix.view();
  par::cuda pf;
  par::seq sf;
  sf.run(hm.shape(), [hm](auto idx) {
    hm(idx) = idx[0] + idx[1] * 2;
  });

  view_copy(m, hm);
  auto mm = eigen_support::cmap(m.as_const());
  pf.run(m.shape(), [mm] __device__(const auto &i) {
    printf("mm(%d, %d) = %f\n", i[0], i[1], mm(i[0], i[1]));
  });

  if (const auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
