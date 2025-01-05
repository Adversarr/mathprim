#include <iostream>
#include <ostream>
#define MATHPRIM_VERBOSE_MALLOC 1
#define MATHPRIM_CPU_BLAS blas
#include "mathprim/core/blas.hpp"
#include "mathprim/core/blas/cpu_handmade.hpp"
#include "mathprim/core/blas/cpu_blas.hpp"
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/buffer_view.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;

#define MATHPRIM_EQUAL(a, b)                         \
  if (std::abs((a) - (b)) > 1e-6) {                  \
    printf("Error " #a "=%f " #b "=%f\n", (a), (b)); \
  }

static constexpr index_t N = 24;

int main() {
  auto x = mathprim::make_buffer<float>(N);
  auto y = mathprim::make_buffer<float>(N);
  auto x_view = x.view();
  auto y_view = y.view();

  for (int i = 0; i < N; i++) {
    x_view[i] = i;
    y_view[i] = N - i;
  }

  blas::scal(2.0f, x_view);
  for (int i = 0; i < N; i++) {
    MATHPRIM_EQUAL(x_view[i], 2.0f * i);
  }

  blas::axpy(1.0f, y_view.as_const(), x_view);
  for (int i = 0; i < N; i++) {
    MATHPRIM_EQUAL(x_view[i], 2.0f * i + N - i);
  }

  blas::copy(y_view, x_view.as_const());
  for (int i = 0; i < N; i++) {
    MATHPRIM_EQUAL(y_view[i], 2.0f * i + N - i);
  }

  const index_t rows = 4, cols = 6;
  auto a = mathprim::make_buffer<float>(rows, cols);
  auto a_view = a.view();
  auto a_1d = a_view.flatten();

  blas::copy(a_1d, y_view.as_const());
  for (auto [i, j] : a.shape()) {
    MATHPRIM_EQUAL(a_view(i, j), 2.0f * (i * cols + j) + N - (i * cols + j));
  }
  memset(a, 1);
  for (auto [i, j] : a.shape()) {
    a_view(i, j) = 1;
  }
  auto a_t = a_view.transpose(-1, -2);

  auto b = mathprim::make_buffer<float>(rows),
       c = mathprim::make_buffer<float>(cols);
  memset(b, 0);
  auto b_view = b.view(), c_view = c.view();
  for (auto i : c.shape()) {
    c_view(i) = 1;
  }
  blas::gemv(1.0f, a_view.as_const(), c.view().as_const(), 0.0f, b_view);
  for (auto i : b_view.shape()) {
    MATHPRIM_EQUAL(b_view(i), 6.0f);
    b_view(i) = 1;
  }

  blas::gemv(1.0f, a_t.as_const(), b.view().as_const(), 0.0f, c.view());
  for (auto i : c_view.shape()) {
    MATHPRIM_EQUAL(c_view(i), 4.0f);
  }

  {
    auto d = mathprim::make_buffer<float>(rows, rows);
    auto d_view = d.view();
    memset(d, 0);
    // d <- a * a_t
    blas::gemm(1.0f, a_view.as_const(), a_t.as_const(), 0.0f, d_view);

    for (auto [i, j] : d.shape()) {
      MATHPRIM_EQUAL(d_view(i, j), 6.0f);
    }
  }

  {
    auto d = mathprim::make_buffer<float>(cols, cols);
    auto d_view = d.view();
    memset(d, 0);
    // dt <- a_t * a
    blas::gemm(1.0f, a_t.as_const(), a_view.as_const(), 0.0f,
               d_view.transpose(-1, -2));
    for (auto [i, j] : d.shape()) {
      MATHPRIM_EQUAL(d_view(i, j), 4.0f);
    }
  }

  return 0;
}
