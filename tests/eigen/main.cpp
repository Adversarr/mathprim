#include <gtest/gtest.h>

#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/view.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/supports/eigen_dense.hpp>
#include <mathprim/supports/eigen_sparse.hpp>

#include "mathprim/blas/cpu_eigen.hpp"
#include "mathprim/linalg/svd.hpp"
#include "mathprim/linalg/inv.hpp"

using namespace mathprim;

GTEST_TEST(matrix, cmap) {
  auto buf = make_buffer<float>(shape_t<4, 3>());
  auto view = buf.view();
  auto map = eigen_support::cmap(view);
  Eigen::Matrix4f m;
  for (auto [c, r] : view.shape()) {
    view(c, r) = static_cast<float>(r * 4 + c);
    m(r, c) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r] : view.shape()) {
    EXPECT_EQ(map(r, c), m(r, c));
  }
}

GTEST_TEST(matrix, map) {
  auto buf = make_buffer<float>(shape_t<4, 3>());
  auto view = buf.view().transpose();
  auto map = eigen_support::map(view);
  Eigen::Matrix4f m;
  for (auto [c, r] : view.shape()) {
    view(c, r) = static_cast<float>(r * 4 + c);
    m(r, c) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r] : view.shape()) {
    EXPECT_EQ(map(r, c), m(r, c));
  }
}

GTEST_TEST(vector, cmap) {
  auto buf = make_buffer<float>(shape_t<4>());
  auto view = buf.view();
  auto map = eigen_support::cmap(view);
  Eigen::Vector4f m;
  for (auto [c] : view.shape()) {
    view(c) = static_cast<float>(c);
    m(c) = static_cast<float>(c);
  }
  for (auto [c] : view.shape()) {
    EXPECT_EQ(map(c), m(c));
  }
}

GTEST_TEST(vector, map) {
  auto buf = make_buffer<float>(shape_t<4>());
  auto view = buf.view();
  auto map = eigen_support::map(view);
  Eigen::Vector4f m;
  for (auto [c] : view.shape()) {
    view(c) = static_cast<float>(c);
    m(c) = static_cast<float>(c);
  }
  for (auto [c] : view.shape()) {
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
  for (auto [c, r] : view.shape()) {
    EXPECT_EQ(view(c, r), m(r, c));
  }
}

GTEST_TEST(matrix, amap) {
  auto buf = make_buffer<float>(shape_t<-1, 4>(3, 4));
  auto view = buf.view();
  auto transposed = view.transpose();
  auto map = eigen_support::amap(view);
  auto tmap = eigen_support::amap(transposed);

  for (auto [c, r] : view.shape()) {
    map(r, c) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r] : view.shape()) {
    EXPECT_EQ(tmap(c, r), static_cast<float>(r * 4 + c));
  }
}

GTEST_TEST(vector, view) {
  Eigen::Vector4f m;
  for (int c = 0; c < 4; ++c) {
    m(c) = static_cast<float>(c);
  }

  auto view = eigen_support::view(m);
  for (auto [c] : view.shape()) {
    EXPECT_EQ(view(c), m(c));
  }

  Eigen::MatrixX3d x3d;
  auto view_x3d = eigen_support::view(x3d);


  Eigen::Matrix3Xd x3d2;
  auto view_x3d2 = eigen_support::view(x3d2);
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

static void ortho(Eigen::Matrix3f& m) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  m = svd.matrixU();
}

GTEST_TEST(svd, 3d_exact) {
  using svd_t = linalg::small_svd<float, device::cpu, 3, 3, true>;
  svd_t svd;
  Eigen::Matrix3f random_mat1 = Eigen::Matrix3f::Random(); ortho(random_mat1);
  Eigen::Matrix3f random_mat2 = Eigen::Matrix3f::Random(); ortho(random_mat2);
  Eigen::Vector3f random_vec = Eigen::Vector3f::Random();

  Eigen::Matrix3f mat = random_mat1 * random_vec.asDiagonal() * random_mat2.transpose();
  Eigen::Matrix3f u, v;
  Eigen::Vector3f sigma;
  svd(eigen_support::view(mat), eigen_support::view(u), eigen_support::view(v), eigen_support::view(sigma));

  auto svd_gt = mat.transpose().eval().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  auto v_gt = svd_gt.matrixV().transpose().eval();
  auto u_gt = svd_gt.matrixU().transpose().eval();
  auto sigma_gt = svd_gt.singularValues().eval();

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(std::abs(u(r, c)), std::abs(u_gt(r, c)), 1e-6);
      EXPECT_NEAR(std::abs(v(r, c)), std::abs(v_gt(r, c)), 1e-6);
    }
    EXPECT_NEAR(sigma(r), sigma_gt(r), 1e-6);
  }

  // // If fails, check these.
  // std::cout << "r1: " << std::endl << random_mat1 << std::endl;
  // std::cout << "r2: " << std::endl << random_mat2 << std::endl;
  // std::cout << "v: " << std::endl << random_vec << std::endl;
  std::cout << "u: " << std::endl << u << std::endl;
  std::cout << "u_gt: " << std::endl << u_gt << std::endl;
  std::cout << "v: " << std::endl << v << std::endl;
  std::cout << "v_gt: " << std::endl << v_gt << std::endl;
  std::cout << "sigma: " << std::endl << sigma << std::endl;
  std::cout << "sigma_gt: " << std::endl << sigma_gt << std::endl;
}

GTEST_TEST(inv, 3d) {
  Eigen::Matrix3f mat = Eigen::Matrix3f::Random();
  ortho(mat);

  Eigen::Matrix3f inv_gt = mat.inverse();
  Eigen::Matrix3f inv;
  using inv_t = linalg::small_inv<float, device::cpu, 3>;
  inv_t inv_op;
  inv_op(eigen_support::view(inv), eigen_support::view(mat));

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(inv(r, c), inv_gt(r, c), 1e-6);
    }
  }
  std::cout << "mat: " << std::endl << mat << std::endl;
  std::cout << "inv: " << std::endl << inv << std::endl;
  std::cout << "inv_gt: " << std::endl << inv_gt << std::endl;
}

GTEST_TEST(view, rect) {
  Eigen::Matrix<float, 4, 3> m;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      m(r, c) = static_cast<float>(r * 4 + c);
    }
  }

  auto view = eigen_support::view(m);
  for (auto [c, r] : view.shape()) {
    EXPECT_EQ(view(c, r), m(r, c));
  }
  using namespace literal;
  auto buf = make_buffer<float>(4_s, 3_s);
  auto view2 = buf.view();
  auto map = eigen_support::map(view2);
  std::cout << map.RowsAtCompileTime << std::endl;
  std::cout << map.ColsAtCompileTime << std::endl;
  for (auto [c, r] : view2.shape()) {
    view2(c, r) = static_cast<float>(r * 4 + c);
  }

  for (auto [c, r] : view2.shape()) {
    EXPECT_EQ(view2(c, r), map(r, c));
  }
}