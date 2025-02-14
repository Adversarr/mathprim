#include <gtest/gtest.h>

#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/supports/eigen_dense.hpp>
#include <mathprim/supports/eigen_sparse.hpp>

#include "mathprim/blas/cpu_eigen.hpp"

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

GTEST_TEST(sparse, csr_rm) {
  Eigen::SparseMatrix<float, Eigen::RowMajor> m(4, 3);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      m.insert(r, c) = static_cast<float>(r * 4 + c);
    }
  }
  m.makeCompressed();

  auto view = eigen_support::view(m);
  EXPECT_EQ(view.rows(), 4);
  EXPECT_EQ(view.cols(), 3);
  EXPECT_EQ(view.nnz(), 12);

  sparse::blas::naive<float, sparse::sparse_format::csr> blas(view.as_const());
  auto x = make_buffer<float>(3);
  auto y = make_buffer<float>(4);
  auto xv = x.view();
  auto yv = y.view();

  auto eigen_x = eigen_support::cmap(xv);
  for (int c = 0; c < 3; ++c) {
    eigen_x(c) = static_cast<float>(c);
  }
  auto eigen_y = eigen_support::cmap(yv);

  blas.gemv(1.0f, xv, 0.0f, yv);
  auto correct = (m * eigen_x).eval();

  for (int c = 0; c < 4; ++c) {
    EXPECT_EQ(eigen_y(c), correct(c));
  }

  sparse::blas::eigen<float, sparse::sparse_format::csr> blas2(view.as_const());
  blas2.gemv(1.0f, xv, 0.0f, yv);
  for (int c = 0; c < 4; ++c) {
    EXPECT_EQ(eigen_y(c), correct(c));
  }
}


GTEST_TEST(sparse, csc_cm) {
  Eigen::SparseMatrix<float, Eigen::ColMajor> m(4, 3);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      m.insert(r, c) = static_cast<float>(r * 4 + c);
    }
  }
  m.makeCompressed();

  auto view = eigen_support::view(m);
  EXPECT_EQ(view.rows(), 4);
  EXPECT_EQ(view.cols(), 3);
  EXPECT_EQ(view.nnz(), 12);

  sparse::blas::naive<float, sparse::sparse_format::csc> blas(view.as_const());
  auto x = make_buffer<float>(3);
  auto y = make_buffer<float>(4);
  auto xv = x.view();
  auto yv = y.view();

  auto eigen_x = eigen_support::cmap(xv);
  for (int c = 0; c < 3; ++c) {
    eigen_x(c) = static_cast<float>(c);
  }
  auto eigen_y = eigen_support::cmap(yv);

  blas.gemv(1.0f, xv, 0.0f, yv);
  auto correct = (m * eigen_x).eval();

  for (int c = 0; c < 4; ++c) {
    EXPECT_EQ(eigen_y(c), correct(c));
  }

  sparse::blas::eigen<float, sparse::sparse_format::csc> blas2(view.as_const());
  blas2.gemv(1.0f, xv, 0.0f, yv);
  for (int c = 0; c < 4; ++c) {
    EXPECT_EQ(eigen_y(c), correct(c));
  }
}