#include <iostream>
#define MATHPRIM_VERBOSE_MALLOC 1
#define MATHPRIM_CPU_BLAS blas
#include <math.h>

#include <mathprim/core/backends/cuda.cuh>
#include <mathprim/core/parallel/cuda.cuh>
#include <mathprim/core/common.hpp>
#include <mathprim/supports/stringify.hpp>

#include "mathprim/core/blas.hpp"
#include "mathprim/core/blas/cublas.cuh"

using namespace mathprim;
static constexpr index_t N = 24;

#define MATHPRIM_EQUAL(a, b)                         \
  if (::abs((a) - (b)) > 1e-6) {                     \
    printf("Error " #a "=%f " #b "=%f\n", (a), (b)); \
  }

__global__ void setup_x(f32_buffer_view<1, device_t::cuda> x) {
  auto i = blockIdx.x;
  x(i) = i;
}

__global__ void setup_y(f32_buffer_view<1, device_t::cuda> y) {
  auto i = blockIdx.x;
  y(i) = N - i;
}

__global__ void check1(f32_buffer_view<1, device_t::cuda> x) {
  auto i = blockIdx.x;
  MATHPRIM_EQUAL(x(i), 2.0f * i);
}

__global__ void check2(f32_buffer_view<1, device_t::cuda> x) {
  auto i = blockIdx.x;
  MATHPRIM_EQUAL(x(i), 2.0f * i + N - i);
}

__global__ void check3(f32_buffer_view<2, device_t::cuda> a) {
  auto i = blockIdx.x;
  auto j = blockIdx.y;
  MATHPRIM_EQUAL(a(i, j),
                 2.0f * (i * a.shape(1) + j) + N - (i * a.shape(1) + j));
}

__global__ void check4(f32_buffer_view<2, device_t::cuda> a) {
  auto i = blockIdx.x;
  auto j = blockIdx.y;
  a(i, j) = 1.0f;
  MATHPRIM_EQUAL(a(i, j), 1.0f);
}

__global__ void ones_(f32_buffer_view<1, device_t::cuda> x) {
  auto i = blockIdx.x;
  x(i) = 1.0f;
}

__global__ void check5(f32_buffer_view<1, device_t::cuda> x) {
  auto i = blockIdx.x;
  MATHPRIM_EQUAL(x(i), 6.0f);
}

int main() {

  auto x = mathprim::make_buffer<float, device_t::cuda>(N);
  auto y = mathprim::make_buffer<float, device_t::cuda>(N);
  auto x_view = x.view();
  auto y_view = y.view();

  dim3 grid_dim = {N, 1, 1};
  dim3 block_dim = {1, 1, 1};

  setup_x<<<grid_dim, block_dim>>>(x_view);
  setup_y<<<grid_dim, block_dim>>>(y_view);
  using blas_ = blas::blas_impl_cublas<float>;
  using parfor_ = parfor<par::cuda>;

  blas_::scal(2.0f, x_view);
  check1<<<grid_dim, block_dim>>>(x_view);

  blas_::axpy(1.0f, y_view.as_const(), x_view);
  check2<<<grid_dim, block_dim>>>(x_view);

  blas_::copy(y_view, x_view.as_const());
  check2<<<grid_dim, block_dim>>>(y_view);

  const index_t rows = 4, cols = 6;
  auto a = mathprim::make_buffer<float, device_t::cuda>(rows, cols);
  auto a_view = a.view();
  auto a_1d = a_view.flatten();

  blas_::copy(a_1d, y_view.as_const());
  // for (auto [i, j] : a.shape()) {
  //   MATHPRIM_EQUAL(a_view(i, j), 2.0f * (i * cols + j) + N - (i * cols +
  //   j));
  // }
  grid_dim = {rows, cols, 1};
  check3<<<grid_dim, block_dim>>>(a_view);
  memset(a, 1);
  // for (auto [i, j] : a.shape()) {
  //   a_view(i, j) = 1;
  // }
  check4<<<grid_dim, block_dim>>>(a_view);
  auto a_t = a_view.transpose(-1, -2);

  auto b = mathprim::make_buffer<float, device_t::cuda>(rows),
       c = mathprim::make_buffer<float, device_t::cuda>(cols);
  memset(b, 0);
  auto b_view = b.view(), c_view = c.view();
  ones_<<<cols, 1>>>(c_view);
  blas_::gemv(1.0f, a_view.as_const(), c.view().as_const(), 0.0f, b_view);
  check5<<<rows, 1>>>(b_view);
  parfor_::for_each(b_view, [] __device__(float &x) { x = 1.0f; });

  blas_::gemv(1.0f, a_t.as_const(), b.view().as_const(), 0.0f, c.view());
  parfor_::for_each(c_view, [] __device__(float &x) { MATHPRIM_EQUAL(x, 4.0f); });

  {
    auto d = mathprim::make_buffer<float, device_t::cuda>(rows, rows);
    auto d_view = d.view();
    memset(d, 0);
    // d <- a * a_t
    blas_::gemm(1.0f, a_view, a_t, 0.0f, d_view);
    parfor_::for_each(d_view, [] __device__(float &x) { MATHPRIM_EQUAL(x, 6.0f); });
  }

  {
    constexpr index_t m = 3, n = 4, k = 5;
    auto a = mathprim::make_buffer<float, device_t::cuda>(m, k);
    auto b = mathprim::make_buffer<float, device_t::cuda>(k, n);
    auto c = mathprim::make_buffer<float, device_t::cuda>(m, n);

    auto a_view = a.view(), b_view = b.view(), c_view = c.view();
    // for (auto [i, j] : a.shape()) {
    //   a_view(i, j) = i * k + j;
    // }
    parfor_::for_each_indexed(a_view, [] __device__(dim<2> ij, float &x) {
      auto [i, j] = ij;
      x = i * k + j;
    });
    // for (auto [i, j] : b.shape()) {
    //   b_view(i, j) = i * n + j;
    // }
    parfor_::for_each_indexed(b_view, [] __device__(dim<2> ij, float &x) {
      auto [i, j] = ij;
      x = i * n + j;
    });
    memset(c, 0);
    blas_::gemm(1.0f, a_view.as_const(), b_view.as_const(), 0.0f, c_view);
    auto c_gt = mathprim::make_buffer<float>(m, n);
    
    memset(c, 0);
    blas_::gemm(1.0f, b_view.transpose(), a_view.transpose(), 0.0f,
                c_view.transpose());
    parfor_::for_each_indexed(c_view, [] __device__(dim<2> ij, float &x) {
      auto [i, j] = ij;
      printf("%d %d %f\n", i, j, x);
    });

    try {
      blas_::gemm(1.0f, a_view.as_const(), b_view.as_const(), 0.0f,
                  c_view.transpose());
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  cudaDeviceSynchronize();
  return 0;
}
