#include <gtest/gtest.h>
#include "mathprim/dnn/nn/linear.hpp"
#include "mathprim/dnn/nn/sequential.hpp"
#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;

constexpr index_t B = 5; // batch_size

GTEST_TEST(nn, linear) {
  dnn::linear<float, device::cpu> lin(3, 2, false);
  dnn::basic_ctx<float, device::cpu, blas::cpu_blas<float>, par::seq, dshape<1>, dshape<1>> ctx;
  ctx.compile(lin, B);
  ctx.zero_grad(lin);

  par::seq().run(lin.mat().shape(), [W = lin.mat()](auto ij) {
    auto [i, j] = ij;
    W(i, j) = i + j;
  });

  auto buf_x = make_buffer<float>(B, 3);
  auto buf_y = make_buffer<float>(B, 2);

  auto x = buf_x.view();
  auto y_true = buf_y.view();
  lin.forward(ctx, x);
  ctx.blas().gemm(1.0, x, lin.mat().transpose(), 0.0, y_true);

  auto y = lin.output();
  for (index_t b = 0; b < B; ++b) {
    for (index_t j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(y(b, j), y_true(b, j));
    }
  }

  auto dl_dy = lin.output_gradient();
  par::seq().run(dl_dy.shape(), [dl_dy](auto ij) {
    auto [i, j] = ij;
    dl_dy(i, j) = i + j;
  });

  // y <- x W.T, w: OUTxIN = 2x3.
  auto dL_dW = ctx.params_gradient().reshape(make_shape(2, 3));
  auto buf_dL_dW_true = make_buffer<float>(2, 3);
  auto dL_dW_true = buf_dL_dW_true.view();
  
  lin.backward(ctx, {});
  // x: [b, in], dl_dy: [b, out], dL_dW: [in, out]
  ctx.blas().gemm(1.0, dl_dy.transpose(), x, 0.0, dL_dW_true);
  for (index_t i = 0; i < 2; ++i) {
    for (index_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(dL_dW(i, j), dL_dW_true(i, j));
    }
  }

  ctx.zero_grad(lin);
  for (index_t i = 0; i < 2; ++i) {
    for (index_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(dL_dW(i, j), 0);
    }
  }

  // call twice
  par::seq().run(dl_dy.shape(), [dl_dy](auto ij) {
    auto [i, j] = ij;
    dl_dy(i, j) = i + j;
  });
  lin.backward(ctx, {});
  for (index_t i = 0; i < 2; ++i) {
    for (index_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(dL_dW(i, j), dL_dW_true(i, j));
    }
  }
}

GTEST_TEST(nn, linear_two) {
  using lin34_t = dnn::linear<float, device::cpu>;
  using lin42_t = dnn::linear<float, device::cpu>;
  dnn::internal::sequential_impl_rec<lin34_t, lin42_t, 0, true> seq(
    lin34_t(3, 4, false),
    lin42_t(4, 2, false)
  );

  auto W1t = eigen_support::cmap(seq.front().mat()); // 3, 4
  auto W2t = eigen_support::cmap(seq.back().mat());  // 4, 2
  W1t.setRandom();
  W2t.setRandom();

  dnn::basic_ctx<float, device::cpu, blas::cpu_blas<float>, par::seq, dshape<1>, dshape<1>> ctx;
  ctx.compile(seq, B);
  ctx.zero_grad(seq);

  auto buf_x = make_buffer<float>(B, 3);
  auto x = eigen_support::cmap(buf_x.view()); // (3, B)
  x.setRandom();

  auto y = seq.forward(ctx, buf_x.view());

  auto y_true = (W2t.transpose() * (W1t.transpose() * x)).eval();

  for (index_t b = 0; b < B; ++b) {
    for (index_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(y(b, j), y_true(j, b), 1e-4);
    }
  }

  auto dL_dW1 = eigen_support::cmap(ctx.params_gradient().sub(0, 12).reshape(make_shape(4, 3)));  // 3, 4
  auto dL_dW2 = eigen_support::cmap(ctx.params_gradient().sub(12, 20).reshape(make_shape(2, 4)));  // 4, 2

  auto dl_dy = eigen_support::cmap(seq.output_gradient());        // (2, B)
  for (auto [i, j]: seq.output_gradient().shape()) {
    seq.output_gradient()(i, j) = i + j;
  }
  auto z = eigen_support::cmap(seq.intermediate());               // (4, B)
  auto dl_dz = eigen_support::cmap(seq.intermediate_gradient());  // (4, B)
  auto dl_dx_buf = make_buffer<float>(B, 3);
  auto dl_dx = dl_dx_buf.view();
  seq.backward(ctx, dl_dx);

  auto dL_dW2_true = (z * dl_dy.transpose()).eval(); // (4, 2)
  auto dl_dz_true = (W2t * dl_dy).eval(); // (4, B)
  auto dL_dW1_true = (x * dl_dz.transpose()).eval(); // (3, 4)
  auto dl_dx_true = (W1t * dl_dz_true).eval(); // (3, B)

  for (index_t i = 0; i < 4; ++i) {
    for (index_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(dL_dW2(i, j), dL_dW2_true(i, j), 1e-4);
    }
  }

  for (index_t i = 0; i < 3; ++i) {
    for (index_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(dL_dW1(i, j), dL_dW1_true(i, j), 1e-4);
    }
  }

  for (index_t i = 0; i < 3; ++i) {
    for (index_t b = 0; b < B; ++b) {
      EXPECT_NEAR(dl_dx(b, i), dl_dx_true(i, b), 1e-4);
    }
  }

  // twice.
  ctx.zero_grad(seq);
  for (auto [i, j]: seq.output_gradient().shape()) {
    seq.output_gradient()(i, j) = i + j;
  }
  seq.backward(ctx, dl_dx);
  for (index_t i = 0; i < 4; ++i) {
    for (index_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(dL_dW2(i, j), dL_dW2_true(i, j), 1e-4);
    }
  }
  for (index_t i = 0; i < 3; ++i) {
    for (index_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(dL_dW1(i, j), dL_dW1_true(i, j), 1e-4);
    }
  }
  for (index_t i = 0; i < 3; ++i) {
    for (index_t b = 0; b < B; ++b) {
      EXPECT_NEAR(dl_dx(b, i), dl_dx_true(i, b), 1e-4);
    }
  }
}

GTEST_TEST(linear, seq) {
  using lin_t = dnn::linear<float, device::cpu>;
  dnn::sequential<lin_t, lin_t, lin_t> seq(
    lin_t(3, 4, false),
    lin_t(4, 5, false),
    lin_t(5, 2, false)
  );

  auto w1 = seq.get<0>().mat();
  auto w2 = seq.get<1>().mat();
  auto w3 = seq.get<2>().mat();

  EXPECT_TRUE(w1.shape() == make_shape(4, 3));
  EXPECT_TRUE(w2.shape() == make_shape(5, 4));
  EXPECT_TRUE(w3.shape() == make_shape(2, 5));
}