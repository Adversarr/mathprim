#include <gtest/gtest.h>
#define MATHPRIM_VERBOSE_MALLOC 1
#include <iostream>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>

using namespace mathprim;

GTEST_TEST(view, iteration) {
  shape_t<keep_dim, 3, 2> shape(4, 3, 2);
  int p[24];
  for (int i = 0; i < 24; ++i) {
    p[i] = i + 1;
  }

  auto v = view<device::cpu>(p, shape);
  auto value0 = v(0, 0, 0);

  auto transpose_view = v.transpose();

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(v(i, j, k), transpose_view(i, k, j));
        EXPECT_EQ(i * 6 + j * 2 + k + 1, v(i, j, k));
      }
    }
  }

  ASSERT_TRUE(v.is_contiguous());
  ASSERT_FALSE(transpose_view.is_contiguous());

  EXPECT_EQ(value0, 1);

  {
    auto stride = make_default_stride(shape);
    auto [i, j, k] = stride;
    EXPECT_EQ(i, 6);
    EXPECT_EQ(j, 2);
    EXPECT_EQ(k, 1);
  }

  {
    auto view2 = v[1];
    auto [j2, k2] = view2.stride();
    EXPECT_EQ(j2, 2);
    EXPECT_EQ(k2, 1);
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(v(1, j, k), view2(j, k));
      }
    }
  }

  {
    auto view2 = v.slice<1>(1);
    auto [j2, k2] = view2.stride();
    EXPECT_EQ(j2, 6);
    EXPECT_EQ(k2, 1);
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(v(i, 1, k), view2(i, k));
      }
    }
  }

  {
    for (auto view2 : v) {
      auto [j2, k2] = view2.stride();
      EXPECT_EQ(j2, 2);
      EXPECT_EQ(k2, 1);
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          EXPECT_EQ(v(0, j, k), view2(j, k));
        }
      }
      break;
    }
  }
}

GTEST_TEST(buffer, creation) {
  {
    auto buf = make_buffer<int>(shape_t<keep_dim, 3, 2>(4, 3, 2));
    auto [i, j, k] = buf.stride();
    EXPECT_EQ(i, 6);
    EXPECT_EQ(j, 2);
    EXPECT_EQ(k, 1);
  }

  {
    using namespace mathprim::literal;
    auto buf = make_buffer<int>(4, 3, 2_s);
    auto [i, j, k] = buf.stride();
    EXPECT_EQ(i, 6);
    EXPECT_EQ(j, 2);
    EXPECT_EQ(k, 1);
  }

  {
    auto buf = make_buffer<int>(make_dshape(4, 3, 2));
    auto view = buf.view();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          buf.data()[i * 6 + j * 2 + k] = i * 6 + j * 2 + k + 1;
          EXPECT_EQ(view(i, j, k), i * 6 + j * 2 + k + 1);
        }
      }
    }

    for (const auto &vi : view) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          EXPECT_EQ(view(0, j, k), j * 2 + k + 1);
        }
      }
      break;
    }
  }
}

GTEST_TEST(blas, handmade) {
  auto buf = make_buffer<float>(make_dshape(4, 3, 2));
  float p[24];
  for (int i = 0; i < 24; ++i) {
    p[i] = static_cast<float>(i + 1);
  }

  blas::cpu_handmade<float> b;
  b.copy(buf.view(), view<device::cpu>(p, make_dshape(4, 3, 2)).as_const());
  for (int i = 0; i < 24; ++i) {
    auto idx = ind2sub(buf.shape(), i);
    EXPECT_EQ(buf.view()(idx), i + 1);
  }
  b.scal(2, buf.view());
  for (int i = 0; i < 24; ++i) {
    auto idx = ind2sub(buf.shape(), i);
    EXPECT_EQ(buf.view()(idx), (i + 1) * 2);
  }

  b.swap(buf.view(), view<device::cpu>(p, make_dshape(4, 3, 2)));
  for (int i = 0; i < 24; ++i) {
    auto idx = ind2sub(buf.shape(), i);
    EXPECT_EQ(buf.view()(idx), i + 1);
  }

  ASSERT_TRUE(blas::internal::is_capable_vector<float>(buf.shape(), buf.stride()));
  ASSERT_FALSE(blas::internal::is_capable_vector<float>(make_dshape(4, 3, 2), make_dshape(24, 16, 4)));

  {
    auto matrix = make_buffer<float>(make_dshape(4, 3));
    auto matrix2 = make_buffer<float>(make_dshape(3, 2));
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix.view()(i, j) = float(i + j);
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        matrix2.view()(i, j) = float(i * 2 + j);
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
  }
}

GTEST_TEST(view, sub) {
  auto buf = make_buffer<int>(make_dshape(4, 3, 2));
  for (auto [i, j, k] : buf.shape()) {
    buf.view()(i, j, k) = i * 6 + j * 2 + k + 1;
  }
  auto bv = buf.view();
  auto sub = bv.sub(index_array<3>{1, 1, 1}, make_dshape(3, 2, 1));
  for (auto [i, j, k] : sub.shape()) {
    EXPECT_EQ(sub(i, j, k), bv(i + 1, j + 1, k + 1));
  }
}

GTEST_TEST(view, sub1d) {
  auto buf = make_buffer<int>(make_dshape(4));
  for (auto i : buf.shape()) {
    buf.view()(i) = i + 1;
  }
  auto bv = buf.view();
  {
    auto sub = bv.sub(index_array<1>{1}, make_dshape(3));
    for (auto i : sub.shape()) {
      EXPECT_EQ(sub(i), bv(i + 1));
    }
  }
  {
    auto sub = bv.sub(1, 4);
    for (auto i : sub.shape()) {
      EXPECT_EQ(sub(i), bv(i + 1));
    }
  }
}

GTEST_TEST(view, mdslice) {
  auto buf = make_buffer<int>(make_dshape(4, 3, 2));
  for (auto [i, j, k] : buf.shape()) {
    buf.view()(i, j, k) = i * 6 + j * 2 + k + 1;
  }
  auto bv = buf.view();
  // auto sub = bv.mdslice<1>(index_array<1>{1});
  {
    auto sub = bv.mdslice(1);
    for (auto [i, j] : sub.shape()) {
      EXPECT_EQ(sub(i, j), bv(1, i, j));
    }
  }
  {
    auto sub = bv.mdslice(1, 2);
    for (index_t i : sub.shape()) {
      EXPECT_EQ(sub(i), bv(1, 2, i));
    }
  }

  {
    auto sub = bv(1, 2);
    for (index_t i : sub.shape()) {
      EXPECT_EQ(sub(i), bv(1, 2, i));
    }
  }
}