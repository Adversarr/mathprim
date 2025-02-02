#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>
#include <mathprim/supports/eigen_dense.hpp>
#include "mathprim/blas/cpu_eigen.hpp"
#include <gtest/gtest.h>

using namespace mathprim;

GTEST_TEST(matrix, cmap) {
  auto buf = make_buffer<float>(shape_t<4, 3>());
  auto view = buf.view();
  auto map = eigen_support::cmap(view);
  Eigen::Matrix4f m;
  for (auto [c, r]: view.shape()) {
    view(c, r) = static_cast<float>(r * 4 + c);
    m(r, c) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r]: view.shape()) {
    EXPECT_EQ(map(r, c), m(r, c));
  }
}

GTEST_TEST(matrix, map) {
  auto buf = make_buffer<float>(shape_t<4, 3>());
  auto view = buf.view().transpose();
  auto map = eigen_support::map(view);
  Eigen::Matrix4f m;
  for (auto [c, r]: view.shape()) {
    view(c, r) = static_cast<float>(r * 4 + c);
    m(r, c) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r]: view.shape()) {
    EXPECT_EQ(map(r, c), m(r, c));
  }
}

GTEST_TEST(vector, cmap) {
  auto buf = make_buffer<float>(shape_t<4>());
  auto view = buf.view();
  auto map = eigen_support::cmap(view);
  Eigen::Vector4f m;
  for (auto [c]: view.shape()) {
    view(c) = static_cast<float>(c);
    m(c) = static_cast<float>(c);
  }
  for (auto [c]: view.shape()) {
    EXPECT_EQ(map(c), m(c));
  }
}

GTEST_TEST(vector, map) {
  auto buf = make_buffer<float>(shape_t<4>());
  auto view = buf.view();
  auto map = eigen_support::map(view);
  Eigen::Vector4f m;
  for (auto [c]: view.shape()) {
    view(c) = static_cast<float>(c);
    m(c) = static_cast<float>(c);
  }
  for (auto [c]: view.shape()) {
    EXPECT_EQ(map(c), m(c));
  }
}

GTEST_TEST(matrix, view) {
  Eigen::Matrix<float, 4, 3> m;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      m(r, c) = static_cast<float>(r * 4 + c);
    }
  }

  auto view = eigen_support::view(m);
  for (auto [c, r]: view.shape()) {
    EXPECT_EQ(view(c, r), m(r, c));
  }
}

GTEST_TEST(matrix, amap) {
  auto buf = make_buffer<float>(shape_t<-1, 4>(3, 4));
  auto view = buf.view();
  auto transposed = view.transpose();
  auto map = eigen_support::amap(view);
  auto tmap = eigen_support::amap(transposed);

  for (auto [c, r]: view.shape()) {
    map(r, c) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r]: view.shape()) {
    EXPECT_EQ(tmap(c, r), static_cast<float>(r * 4 + c));
  }
}

GTEST_TEST(vector, view) {
  Eigen::Vector4f m;
  for (int c = 0; c < 4; ++c) {
    m(c) = static_cast<float>(c);
  }

  auto view = eigen_support::view(m);
  for (auto [c]: view.shape()) {
    EXPECT_EQ(view(c), m(c));
  }
}