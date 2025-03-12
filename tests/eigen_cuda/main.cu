#include "mathprim/linalg/inv.hpp"
#include "mathprim/linalg/svd.hpp"
#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/eigen_dense.hpp>

using namespace mathprim;

int main() {
  auto matrix = make_buffer<float, device::cuda>(4, 4);
  auto mat2 = make_buffer<half, device::cuda>(4, 4);
  auto host_matrix = make_buffer<float>(4, 4);
  auto m = matrix.view();
  auto hm = host_matrix.view();
  par::cuda pf;
  par::seq sf;
  sf.run(hm.shape(), [hm](auto idx) {
    hm(idx) = idx[0] + idx[1] * 2;
  });

  copy(m, hm);
  auto mm = eigen_support::cmap(m.as_const());
  pf.run(m.shape(), [mm] __device__(const auto &i) {
    printf("mm(%d, %d) = %f\n", i[0], i[1], mm(i[0], i[1]));
  });

  if (const auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }
  using namespace literal;
  index_t N = 10;
  auto m33 = make_buffer<float, device::cuda>(N, 3_s, 3_s);
  auto inv33 = make_buffer<float, device::cuda>(N, 3_s, 3_s);
  using inversion = linalg::small_inv<float, device::cuda, 3>;

  pf.run(make_shape(10), [m33 = m33.view()] __device__(const index_t &idx) {
    // init to [2, 1, 0] [1, 2, 1] [0, 1, 2]
    for (index_t i = 0; i < 3; ++i) {
      for (index_t j = 0; j < 3; ++j) {
        m33(idx, i, j) = 2 - std::abs(i - j);
      }
    }
  });

  inv33.fill_bytes(0);

  pf.vmap(inversion(), inv33.view(), m33.view());

  auto inv33_host = make_buffer<float>(N, 3_s, 3_s);
  copy(inv33_host.view(), inv33);
  auto inv33_host_eigen = eigen_support::cmap(inv33_host.view()[0].as_const());
  std::cout << inv33_host_eigen << std::endl;
  Eigen::Matrix3f m33_eigen{
      {2, 1, 0},
      {1, 2, 1},
      {0, 1, 2}
  };

  std::cout << m33_eigen.inverse() << std::endl;

  auto svd_u = make_buffer<float, device::cuda>(N, 3_s, 3_s);
  auto svd_vt = make_buffer<float, device::cuda>(N, 3_s, 3_s);
  auto svd_s = make_buffer<float, device::cuda>(N, 3_s);
  using svd = linalg::small_svd<float, device::cuda, 3, 3>;

  pf.vmap(svd(true), m33.const_view(), svd_u.view(), svd_vt.view(), svd_s.view());

  auto svd_u_host = make_buffer<float>(N, 3_s, 3_s);
  copy(svd_u_host.view(), svd_u);

  auto u0 = eigen_support::cmap(svd_u_host.view()[0].as_const());
  std::cout << u0 << std::endl;

  auto svd_host = m33_eigen.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  std::cout << svd_host.matrixU() << std::endl;

  auto det = make_cuda_buffer<float>(N);
  auto det_host = make_buffer<float>(N);
  using det_t = linalg::small_det<float, device::cuda, 3>;

  pf.vmap(par::make_output_vmapped(det_t()), det.view(), m33.const_view());
  copy(det_host.view(), det);
  std::cout << det_host.data()[0] << std::endl;

  return EXIT_SUCCESS;
}
