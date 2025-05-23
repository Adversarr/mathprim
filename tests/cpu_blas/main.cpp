#include <gtest/gtest.h>

#include <mathprim/core/buffer.hpp>

#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/blas/cpu_handmade.hpp"
#include "mathprim/parallel/parallel.hpp"
using namespace mathprim;

GTEST_TEST(blas, gemv) {
  blas::cpu_blas<float> bb;
  blas::cpu_handmade<float> b;
  blas::cpu_eigen<float> be;
  auto Ab = make_buffer<float>(make_dshape(4, 3));
  auto xb = make_buffer<float>(make_dshape(3));
  auto wb = make_buffer<float>(make_dshape(3));
  auto yb = make_buffer<float>(make_dshape(4));
  auto zb = make_buffer<float>(make_dshape(4));
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
  be.gemv(1.0f, a.as_const(), x.as_const(), 0.0f, z);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(y(i), z(i));
  }

  bb.gemv(1.0f, a.as_const().transpose(), y.as_const(), 0.0f, x);
  b.gemv(1.0f, a.as_const().transpose(), y.as_const(), 0.0f, w);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(x(i), w(i));
  }

  be.gemv(1.0f, a.as_const().transpose(), y.as_const(), 0.0f, w);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(x(i), w(i));
  }
}

GTEST_TEST(blas, gemm) {
  blas::cpu_blas<float> b;
  blas::cpu_eigen<float> be;
  auto matrix = make_buffer<float>(make_dshape(4, 3));
  auto matrix2 = make_buffer<float>(make_dshape(3, 2));
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

  auto out = make_buffer<float>(make_dshape(4, 2));
  auto out_view = out.view();
  b.gemm(1.0f, matrix.const_view(), matrix2.const_view(), 0.0f, out_view);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(out_view(i, j), hand[i][j]);
    }
  }
  be.gemm(1.0f, matrix.const_view(), matrix2.const_view(), 0.0f, out_view);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(out_view(i, j), hand[i][j]);
    }
  }

  b.gemm(1.0f, matrix2.const_view().transpose(), matrix.const_view().transpose(), 0.0f, out_view.transpose());
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(out_view(i, j), hand[i][j]);
    }
  }

  be.gemm(1.0f, matrix2.const_view().transpose(), matrix.const_view().transpose(), 0.0f, out_view.transpose());
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(out_view(i, j), hand[i][j]);
    }
  }
}

GTEST_TEST(blas, inplace_emul) {
  blas::cpu_blas<float> b;
  auto a = make_buffer<float>(12);
  auto x = make_buffer<float>(12);
  auto y = make_buffer<float>(12);
  par::seq().run(a.shape(), [av = a.view(), xv = x.view()](auto i) {
    av(i) = i[0];
    xv(i) = i[0];
  });
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(x.view()[i], i);
  }
  // y <- a y
  b.copy(y.view(), x.const_view());
  b.inplace_emul(a.const_view(), y.view());
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(y.view()[i], i * i);
  }
}

GTEST_TEST(blas, batch_gemm) {
  blas::cpu_blas<float> bl;
  index_t batch = 4;
  index_t m = 3, n = 2, k = 4;
  auto a = make_buffer<float>(batch, m, k);
  auto b = make_buffer<float>(batch, k, n);
  auto c = make_buffer<float>(batch, m, n);
  auto alpha = make_buffer<float>(batch);
  auto beta = make_buffer<float>(batch);
  for (index_t i = 0; i < batch; ++i) {
    for (index_t j = 0; j < m; ++j) {
      for (index_t l = 0; l < k; ++l) {
        a.view()(i, j, l) = static_cast<float>(i + j + l);
      }
    }

    for (index_t j = 0; j < k; ++j) {
      for (index_t l = 0; l < n; ++l) {
        b.view()(i, j, l) = static_cast<float>(i + j + l);
      }
    }

    alpha.view()[i] = 1;
    beta.view()[i] = 0;
  }

  bl.gemm_batch_strided(1.0f, a.const_view(), b.const_view(), 0.0f, c.view());

  for (index_t i = 0; i < batch; ++i) {
    bl.gemm(1.0f, a.const_view().slice(i), b.const_view().slice(i), -1.0f, c.view().slice(i));
  }

  // test norm:
  EXPECT_FLOAT_EQ(0.0f, bl.norm(c.const_view()));
}
