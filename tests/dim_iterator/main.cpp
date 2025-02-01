#include <gtest/gtest.h>
#define MATHPRIM_VERBOSE_MALLOC 1
#include <iostream>
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>

using namespace mathprim;

GTEST_TEST(view, iteration) {
  shape_t<-1, 3, 2> shape(4, 3, 2);
  int p[24];
  for (int i = 0; i < 24; ++i) {
    p[i] = i + 1;
  }

  auto view = make_view<device::cpu>(p, shape);
  auto value0 = view(0, 0, 0);

  auto t = transpose<1, 2>(shape);
  auto transpose_view = view.transpose();

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        ASSERT_EQ(view(i, j, k), transpose_view(i, k, j));
        ASSERT_EQ(i * 6 + j * 2 + k + 1, view(i, j, k));
      }
    }
  }

  ASSERT_TRUE(view.is_contiguous());
  ASSERT_FALSE(transpose_view.is_contiguous());

  ASSERT_EQ(value0, 1);

  {
    auto stride = make_default_stride<int>(shape);
    auto [i, j, k] = stride;
    std::cout << i << " " << j << " " << k << std::endl;
    ASSERT_EQ(i, 24);
    ASSERT_EQ(j, 8);
    ASSERT_EQ(k, 4);
  }

  {
    auto view2 = view[1];
    auto [j2, k2] = view2.stride();
    ASSERT_EQ(j2, 8);
    ASSERT_EQ(k2, 4);
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        ASSERT_EQ(view(1, j, k), view2(j, k));
      }
    }
  }

  {
    auto view2 = view.slice<1>(1);
    auto [j2, k2] = view2.stride();
    ASSERT_EQ(j2, 24);
    ASSERT_EQ(k2, 4);
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 2; ++k) {
        ASSERT_EQ(view(i, 1, k), view2(i, k));
      }
    }
  }

  {
    for (auto view2 : view) {
      auto [j2, k2] = view2.stride();
      ASSERT_EQ(j2, 8);
      ASSERT_EQ(k2, 4);
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          ASSERT_EQ(view(0, j, k), view2(j, k));
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
    ASSERT_EQ(i, 24);
    ASSERT_EQ(j, 8);
    ASSERT_EQ(k, 4);
  }

  {
    auto buf = make_buffer<int>(make_dynamic_shape(4, 3, 2));
    auto [i, j, k] = buf.stride();
    ASSERT_EQ(i, 24);
    ASSERT_EQ(j, 8);
    ASSERT_EQ(k, 4);
  }

  {
    auto buf = make_buffer<int>(make_dynamic_shape(4, 3, 2));
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
          ASSERT_EQ(view(0, j, k), j * 2 + k + 1);
        }
      }
      break;
    }
  }
}

GTEST_TEST(blas, handmade) {
  auto buf = make_buffer<float>(make_dynamic_shape(4, 3, 2));
  float p[24];
  for (int i = 0; i < 24; ++i) {
    p[i] = static_cast<float>(i + 1);
  }

  blas::cpu_handmade<float> b;
  b.copy(buf.view(), make_view<device::cpu>(p, make_dynamic_shape(4, 3, 2)).as_const());
  for (int i = 0; i < 24; ++i) {
    auto idx = ind2sub(buf.shape(), i);
    ASSERT_EQ(buf.view()(idx), i + 1);
  }
  b.scal(2, buf.view());
  for (int i = 0; i < 24; ++i) {
    auto idx = ind2sub(buf.shape(), i);
    ASSERT_EQ(buf.view()(idx), (i + 1) * 2);
  }

  b.swap(buf.view(), make_view<device::cpu>(p, make_dynamic_shape(4, 3, 2)));
  for (int i = 0; i < 24; ++i) {
    auto idx = ind2sub(buf.shape(), i);
    ASSERT_EQ(buf.view()(idx), i + 1);
  }

  ASSERT_TRUE(blas::internal::is_capable_vector<float>(buf.shape(), buf.stride()));
  ASSERT_FALSE(blas::internal::is_capable_vector<float>(make_dynamic_shape(4, 3, 2), make_dynamic_shape(24, 16, 4)));

  {
    auto matrix = make_buffer<float>(make_dynamic_shape(4, 3));
    auto matrix2 = make_buffer<float>(make_dynamic_shape(3, 2));
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

    auto out = make_buffer<float>(make_dynamic_shape(4, 2));
    auto out_view = out.view();
    b.gemm(1.0f, matrix.const_view(), matrix2.const_view(),
           0.0f, out_view);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j) {
        ASSERT_EQ(out_view(i, j), hand[i][j]);
      }
    }
  }
}
