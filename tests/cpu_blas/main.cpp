#define MATHPRIM_VERBOSE_MALLOC 1
#define MATHPRIM_CPU_BLAS blas
#include <gtest/gtest.h>

#include <mathprim/core/buffer.hpp>

#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/blas/cpu_handmade.hpp"
using namespace mathprim;

GTEST_TEST(blas, gemv) {
  blas::cpu_blas<float> bb;
  blas::cpu_handmade<float> b;
  auto Ab = make_buffer<float>(make_dynamic_shape(4, 3));
  auto xb = make_buffer<float>(make_dynamic_shape(3));
  auto wb = make_buffer<float>(make_dynamic_shape(3));
  auto yb = make_buffer<float>(make_dynamic_shape(4));
  auto zb = make_buffer<float>(make_dynamic_shape(4));
  auto a = Ab.view();
  auto x = xb.view();
  auto y = yb.view();
  auto z = zb.view();
  auto w = wb.view();

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      a(i, j) = static_cast<float>(i + j);
    }
  }

  for (int i = 0; i < 3; ++i) {
    x(i) = static_cast<float>(i + 1);
  }

  bb.gemv(1.0f, a.as_const(), x.as_const(), 0.0f, y);
  b.gemv(1.0f, a.as_const(), x.as_const(), 0.0f, z);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(y(i), z(i));
  }

  bb.gemv(1.0f, a.as_const().transpose(), y.as_const(), 0.0f, x);
  b.gemv(1.0f, a.as_const().transpose(), y.as_const(), 0.0f, w);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(x(i), w(i));
  }
}

GTEST_TEST(blas, gemm) {
  blas::cpu_blas<float> b;
  auto matrix = make_buffer<float>(make_dynamic_shape(4, 3));
  auto matrix2 = make_buffer<float>(make_dynamic_shape(3, 2));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      matrix.view()(i, j) = i + j;
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      matrix2.view()(i, j) = i * 2 + j;
    }
  }

  float hand[4][2];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      hand[i][j] = 0;
      for (int k = 0; k < 3; ++k) {
        hand[i][j] += matrix.view()(i, k) * matrix2.view()(k, j);
      }
    }
  }

  auto out = make_buffer<float>(make_dynamic_shape(4, 2));
  auto out_view = out.view();
  b.gemm(1.0f, matrix.const_view(), matrix2.const_view(),
         0.0f, out_view);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(out_view(i, j), hand[i][j]);
    }
  }

  b.gemm(1.0f, matrix2.const_view().transpose(), matrix.const_view().transpose(),
         0.0f, out_view.transpose());
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(out_view(i, j), hand[i][j]);
    }
  }
}
