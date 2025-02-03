#include <math.h>

#include "mathprim/blas/cublas.cuh"
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/parallel/cuda.cuh"

using namespace mathprim;

int main() {
  auto m34 = make_buffer<float, device::cuda>(shape_t<-1, 4>(3, 4));
  auto m23 = make_buffer<float, device::cuda>(shape_t<-1, 3>(2, 3));
  auto m42 = make_buffer<float, device::cuda>(shape_t<-1, 2>(4, 2));
  auto v3 = make_buffer<float, device::cuda>(shape_t<3>(3));
  auto v2 = make_buffer<float, device::cuda>(shape_t<2>(2));
  auto a = m34.view();
  auto b = m23.view();
  auto c = m42.view();
  auto x = v3.view();
  auto y = v2.view();

  blas::cublas<float> bl;
  par::cuda p;
  p.run(a.shape(), [a] __device__ (auto idx) {
    auto [i, j] = idx;
    a(idx) = i + j;
    printf("a(%d, %d) = %f\n", i, j, a(idx));
  });
  p.run(b.shape(), [b] __device__ (auto idx) {
    auto [i, j] = idx;
    b(idx) = i + j;
    printf("b(%d, %d) = %f\n", i, j, b(idx));
  });
  p.run(c.shape(), [c] __device__ (auto idx) {
    auto [i, j] = idx;
    c(idx) = i + j;
    printf("c(%d, %d) = %f\n", i, j, c(idx));
  });

  p.run(x.shape(), [x] __device__ (auto idx) {
    auto [i] = idx;
    x(idx) = i + 1;
    printf("x(%d) = %f\n", i, x(idx));
  });
  p.run(y.shape(), [y] __device__ (auto idx) {
    auto [i] = idx;
    y(idx) = i + 1;
    printf("y(%d) = %f\n", i, y(idx));
  });

  bl.gemv(1.0, b.as_const(), x.as_const(), 0.0, y);

  // A: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
  // B: [[0, 1, 2], [1, 2, 3]]
  // C: [[0, 1], [1, 2], [2, 3], [3, 4]]
  // X: [1, 2, 3]

  // Y = B * X = [[0, 1, 2], [1, 2, 3]] * [1, 2, 3] = [8, 14]
  p.run(y.shape(), [y] __device__ (auto idx) {
    auto [i] = idx;
    printf("%d: %f\n", i, y(idx));
    y(idx) = i + 1; // Y <- [1, 2]
  });

  // X = B.T * Y = [[0, 1, 2], [1, 2, 3]].T * [1, 2] = [2, 5, 8]
  bl.gemv(1.0, b.as_const().transpose(), y.as_const(), 0.0, x);
  p.run(x.shape(), [x] __device__ (auto idx) {
    auto [i] = idx;
    printf("%d: %f\n", i, x(idx));
    x(idx) = i + 1; // X <- [1, 2, 3]
  });

  // C.T = B * A = [[0, 1, 2], [1, 2, 3]] * [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]] = [[5, 8, 11, 14], [8, 14, 20, 26]]
  bl.gemm(1.0, b.as_const(), a.as_const(), 0.0, c.transpose());
  p.run(c.shape(), [c] __device__ (auto idx) {
    auto [i, j] = idx;
    printf("(%d, %d): %f\n", i, j, c(idx));
    c(idx) = i + j; // C.T <- [[0, 1], [1, 2], [2, 3], [3, 4]]
  });

  auto m43 = make_buffer<float, device::cuda>(shape_t<-1, -1>(4, 3));
  auto d = m43.view();
  // D = C * B = [[0, 1], [1, 2], [2, 3], [3, 4]] * [[0, 1, 2], [1, 2, 3]] = [[1, 2, 3], [2, 5, 8], [3, 8, 13], [4, 11, 18]]
  bl.gemm(1.0, c.as_const(), b.as_const(), 0.0, d);
  p.run(d.shape(), [d] __device__ (auto idx) {
    auto [i, j] = idx;
    printf("(%d, %d): %f\n", i, j, d(idx));
  });


  return EXIT_SUCCESS;
}
