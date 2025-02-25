#include <math.h>

#include "mathprim/blas/cpu_blas.hpp"
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

  // Do batched.
  index_t batch = 4;
  index_t m = 3, n = 2, k = 4;
  auto h_a = make_buffer<float>(batch, m, k);
  auto h_b = make_buffer<float>(batch, k, n);
  auto h_c = make_buffer<float>(batch, m, n);
  auto alpha = make_buffer<float>(batch);
  auto beta = make_buffer<float>(batch);
  for (index_t i = 0; i < batch; ++i) {
    for (index_t j = 0; j < m; ++j) {
      for (index_t l = 0; l < k; ++l) {
        h_a.view()(i, j, l) = static_cast<float>(i + j + l);
      }
    }

    for (index_t j = 0; j < k; ++j) {
      for (index_t l = 0; l < n; ++l) {
        h_b.view()(i, j, l) = static_cast<float>(i + j + l);
      }
    }

    alpha.view()[i] = 1;
    beta.view()[i] = 0;
  }

  auto d_a = make_cuda_buffer<float>(batch, m, k);
  auto d_b = make_cuda_buffer<float>(batch, k, n);
  auto d_c = make_cuda_buffer<float>(batch, m, n);

  blas::cpu_blas<float> bl2;
  for (index_t i = 0; i < batch; ++i) {
    bl2.gemm(1.0f, h_a.const_view().slice(i), h_b.const_view().slice(i), 0.0, h_c.view().slice(i));
  }

  copy(d_a.view(), d_a.view());
  copy(d_b.view(), d_b.view());
  copy(d_c.view(), d_c.view());

  bl.gemm_batch_strided(1.0f, d_a.const_view(), d_b.const_view(), -1.0f, d_c.view());
  // assert d_c = 0
  auto norm = bl.norm(d_c.const_view());
  std::cout << norm << std::endl;

  return EXIT_SUCCESS;
}
