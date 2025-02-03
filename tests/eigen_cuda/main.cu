#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>
#include <mathprim/parallel/cuda.cuh>
#include <mathprim/supports/eigen_dense.hpp>

using namespace mathprim;

int main() {
  auto matrix = make_buffer<float, device_t::cuda>(4, 4);
  auto mv = matrix.view();

  parfor_cuda::run(dim_t{mv.shape()}, [mv] __device__(const dim_t &i) {
    mv(dim<2>(i)) = i[0] * 4 + i[1];
  });

  auto mm = eigen_support::map(mv.as_const());
  parfor<par::cuda>::run(dim_t(mv.shape()), [mm] __device__(const dim_t &i) {
    printf("mm(%d, %d) = %f\n", i[0], i[1], mm(i[0], i[1]));
  });

  auto vector = make_buffer<float, device_t::cuda>(4);
  auto vv = vector.view();
  auto map_to_vector = eigen_support::cmap<4>(vv.as_const());

  parfor_cuda::run(dim_t{vv.shape()}, [vv] __device__(const dim_t &i) {
    auto ix = i.x_;
    vv(ix) = ix;
  });

  parfor_cuda::run(dim_t{vv.shape()}, [map_to_vector] __device__(const dim_t &i) {
    printf("map_to_vector(%d) = %f\n", i[0], map_to_vector(i[0]));
  });

  // Another way, use map, ignores the continuous property.
  auto map_to_matrix2 = eigen_support::map<4, 4>(mv.as_const());
  parfor_cuda::run(dim_t{mv.shape()}, [map_to_matrix2] __device__(const dim_t &i) {
    printf("map_to_matrix2(%d, %d) = %f\n", i[0], i[1], map_to_matrix2(i[0], i[1]));
  });

  auto map_to_vector2 = eigen_support::map<4>(vv.as_const());
  parfor_cuda::run(dim_t{vv.shape()}, [map_to_vector2] __device__(const dim_t &i) {
    printf("map_to_vector2(%d) = %f\n", i[0], map_to_vector2(i[0]));
  });

  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
    return 1;
  }

  return 0;
}
