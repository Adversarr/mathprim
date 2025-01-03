#define MATHPRIM_VERBOSE_MALLOC 1
#define MATHPRIM_CPU_BLAS blas
#include "mathprim/core/blas.hpp"
#include "mathprim/core/blas/cublas.cuh"
#include <mathprim/core/backends/cuda.cuh>
#include <mathprim/core/common.hpp>
#include <mathprim/supports/stringify.hpp>

#include <math.h>

using namespace mathprim;
static constexpr index_t N = 24;

#define MATHPRIM_EQUAL(a, b)                                                   \
  if (::abs((a) - (b)) > 1e-6) {                                               \
    printf("Error " #a "=%f " #b "=%f\n", (a), (b));                           \
  }
// } else {                                                                     \
  //   printf("Success " #a "=%f " #b "=%f\n", (a), (b));                         \
  // }

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

  blas::scal(2.0f, x_view);
  check1<<<grid_dim, block_dim>>>(x_view);

  blas::axpy(1.0f, y_view.as_const(), x_view);
  check2<<<grid_dim, block_dim>>>(x_view);

  blas::copy(y_view, x_view.as_const());
  check2<<<grid_dim, block_dim>>>(y_view);

  const index_t rows = 4, cols = 6;
  auto a = mathprim::make_buffer<float, device_t::cuda>(rows, cols);
  auto a_view = a.view();
  auto a_1d = a_view.flatten();

  blas::copy(a_1d, y_view.as_const());
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
  blas::gemv(1.0f, a_view.as_const(), c.view().as_const(), 0.0f, b_view);
  check5<<<rows, 1>>>(b_view);
  // for (auto i : b_view.shape()) {
  //   MATHPRIM_EQUAL(b_view(i), 6.0f);
  //   b_view(i) = 1;
  // }

  // blas::gemv(1.0f, a_t.as_const(), b.view().as_const(), 0.0f, c.view());
  // for (auto i : c_view.shape()) {
  //   MATHPRIM_EQUAL(c_view(i), 4.0f);
  // }

  //   {
  //     auto d = mathprim::make_buffer<float, device_t::cuda>(rows, rows);
  //     auto d_view = d.view();
  //     memset(d, 0);
  //     // d <- a * a_t
  //     blas::gemm(1.0f, a_view.as_const(), a_t.as_const(), 0.0f, d_view);

  //     for (auto [i, j] : d.shape()) {
  //       MATHPRIM_EQUAL(d_view(i, j), 6.0f);
  //     }
  //   }

  //   {
  //     auto d = mathprim::make_buffer<float, device_t::cuda>(cols, cols);
  //     auto d_view = d.view();
  //     memset(d, 0);
  //     // dt <- a_t * a
  //     blas::gemm(1.0f, a_t.as_const(), a_view.as_const(), 0.0f,
  //                d_view.transpose(-1, -2));
  //     for (auto [i, j] : d.shape()) {
  //       MATHPRIM_EQUAL(d_view(i, j), 4.0f);
  //     }
  //   }

  cudaDeviceSynchronize();
  return 0;
}
