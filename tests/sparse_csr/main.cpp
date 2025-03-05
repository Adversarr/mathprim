#include <gtest/gtest.h>

#include <fstream>
#include <mathprim/sparse/blas/cholmod.hpp>
#include <mathprim/sparse/blas/naive.hpp>
#include <mathprim/sparse/systems/laplace.hpp>

#include "mathprim/parallel/openmp.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "mathprim/sparse/cvt.hpp"
#include "mathprim/supports/eigen_sparse.hpp"
#include "mathprim/supports/io/matrix_market.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;
GTEST_TEST(csr, gemv) {
  const int rows = 3, cols = 3, nnz = 5;
  float h_csr_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int h_csr_col_idx[] = {0, 1, 1, 2, 2};
  int h_csr_row_ptr[] = {0, 2, 4, 5};

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      view(h_csr_values, make_shape(nnz)).as_const(), view(h_csr_row_ptr, make_shape(rows + 1)).as_const(),
      view(h_csr_col_idx, make_shape(nnz)).as_const(), rows, cols, nnz, sparse::sparse_property::general);

  float h_x[] = {1.0f, 1.0f, 1.0f};
  float h_y[rows] = {0.0f};
  auto x = view(h_x, make_shape(cols));
  auto y = view(h_y, make_shape(rows));

  {
    sparse::blas::naive<float, sparse::sparse_format::csr> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y);

    // y[0] = 1*1 + 2*1 = 3.00
    // y[1] = 3*1 + 4*1 = 7.00
    // y[2] = 5*1 = 5.00
    EXPECT_FLOAT_EQ(h_y[0], 3.0f);
    EXPECT_FLOAT_EQ(h_y[1], 7.0f);
    EXPECT_FLOAT_EQ(h_y[2], 5.0f);
  }

  {
    sparse::blas::naive<float, sparse::sparse_format::csr, par::openmp> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y);

    // y[0] = 1*1 + 2*1 = 3.00
    // y[1] = 3*1 + 4*1 = 7.00
    // y[2] = 5*1 = 5.00
    EXPECT_FLOAT_EQ(h_y[0], 3.0f);
    EXPECT_FLOAT_EQ(h_y[1], 7.0f);
    EXPECT_FLOAT_EQ(h_y[2], 5.0f);
  }

  {
    sparse::blas::cholmod<float, sparse::sparse_format::csr> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y);
    EXPECT_FLOAT_EQ(h_y[0], 3.0f);
    EXPECT_FLOAT_EQ(h_y[1], 7.0f);
    EXPECT_FLOAT_EQ(h_y[2], 5.0f);
  }
}

GTEST_TEST(csr, gemv_trans) {
  const int rows = 3, cols = 3, nnz = 5;
  float h_csr_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int h_csr_col_idx[] = {0, 1, 1, 2, 2};
  int h_csr_row_ptr[] = {0, 2, 4, 5};
  // [[1, 2, 0],
  //  [0, 3, 4],
  //  [0, 0, 5]]

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      h_csr_values, h_csr_row_ptr, h_csr_col_idx, rows, cols, nnz, sparse::sparse_property::general);

  // [[1, 0, 0],
  //  [2, 3, 0],
  //  [0, 4, 5]]

  float h_x[] = {1.0f, 1.0f, 1.0f};
  float h_y[rows] = {0.0f};
  auto x = view(h_x, make_shape(cols));
  auto y = view(h_y, make_shape(rows));

  {
    sparse::blas::naive<float, sparse::sparse_format::csr> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y, true);

    // y[0] = 1*1 = 1.00
    // y[1] = 2*1 + 3*1 = 5.00
    // y[2] = 4*1 + 5*1 = 9.00
    EXPECT_FLOAT_EQ(h_y[0], 1.0f);
    EXPECT_FLOAT_EQ(h_y[1], 5.0f);
    EXPECT_FLOAT_EQ(h_y[2], 9.0f);
  }

  {
    sparse::blas::naive<float, sparse::sparse_format::csr, par::openmp> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y, true);
    EXPECT_FLOAT_EQ(h_y[0], 1.0f);
    EXPECT_FLOAT_EQ(h_y[1], 5.0f);
    EXPECT_FLOAT_EQ(h_y[2], 9.0f);
  }

  {
    sparse::blas::cholmod<float, sparse::sparse_format::csr> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y, true);
    EXPECT_FLOAT_EQ(h_y[0], 1.0f);
    EXPECT_FLOAT_EQ(h_y[1], 5.0f);
    EXPECT_FLOAT_EQ(h_y[2], 9.0f);
  }
}

GTEST_TEST(csr, convert) {
  const int rows = 3, cols = 3, nnz = 5;
  float h_csr_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int h_csr_col_idx[] = {0, 1, 1, 2, 2};
  int h_csr_row_ptr[] = {0, 2, 4, 5};
  // [[1, 2, 0],
  //  [0, 3, 4],
  //  [0, 0, 5]]

  float h_coo_values[nnz] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int h_coo_row_idx[nnz] = {0, 0, 1, 1, 2};
  int h_coo_col_idx[nnz] = {0, 1, 1, 2, 2};

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> csr(
      h_csr_values, h_csr_row_ptr, h_csr_col_idx, rows, cols, nnz, sparse::sparse_property::general);

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::coo> coo(
      h_coo_values, h_coo_row_idx, h_coo_col_idx, rows, cols, nnz, sparse::sparse_property::general);

  sparse::format_convert<float, sparse::sparse_format::csr, device::cpu> convert;
  {
    auto csr_mat = convert.from_coo(coo);
    EXPECT_EQ(csr_mat.nnz(), csr.nnz());
    auto outer = csr_mat.outer_ptrs().view();
    auto inner = csr_mat.inner_indices().view();
    auto values = csr_mat.values().view();
    for (int i = 0; i <= rows; ++i) {
      EXPECT_EQ(h_csr_row_ptr[i], outer[i]);
    }
    for (int i = 0; i < csr_mat.nnz(); ++i) {
      EXPECT_EQ(inner[i], h_csr_col_idx[i]);
      EXPECT_FLOAT_EQ(values[i], h_csr_values[i]);
    }
  }

  {
    auto coo_mat = convert.to_coo(csr);
    EXPECT_EQ(coo_mat.nnz(), coo.nnz());
    auto row = coo_mat.outer_ptrs().view();
    auto col = coo_mat.inner_indices().view();
    auto values = coo_mat.values().view();
    for (int i = 0; i < coo_mat.nnz(); ++i) {
      EXPECT_EQ(row[i], h_coo_row_idx[i]);
      EXPECT_EQ(col[i], h_coo_col_idx[i]);
      EXPECT_FLOAT_EQ(values[i], h_coo_values[i]);
    }
  }
}

GTEST_TEST(csc, convert) {
  const int rows = 3, cols = 3, nnz = 5;
  float h_csc_values[] = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f};
  int h_csc_row_idx[] = {0, 1, 0, 1, 2};
  int h_csc_col_ptr[] = {0, 2, 4, 5};

  float h_coo_values[nnz] = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f};
  int h_coo_row_idx[nnz] = {0, 1, 0, 1, 2};
  int h_coo_col_idx[nnz] = {0, 0, 1, 1, 2};

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csc> csc(
      h_csc_values, h_csc_col_ptr, h_csc_row_idx, rows, cols, nnz, sparse::sparse_property::general);

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::coo> coo(
      h_coo_values, h_coo_row_idx, h_coo_col_idx, rows, cols, nnz, sparse::sparse_property::general);

  sparse::format_convert<float, sparse::sparse_format::csc, device::cpu> convert;

  {
    auto csc_mat = convert.from_coo(coo);
    EXPECT_EQ(csc_mat.nnz(), csc.nnz());
    auto outer = csc_mat.outer_ptrs().view();
    auto inner = csc_mat.inner_indices().view();
    auto values = csc_mat.values().view();
    for (int i = 0; i <= cols; ++i) {
      EXPECT_EQ(h_csc_col_ptr[i], outer[i]);
    }
    for (int i = 0; i < csc_mat.nnz(); ++i) {
      EXPECT_EQ(inner[i], h_csc_row_idx[i]);
      EXPECT_FLOAT_EQ(values[i], h_csc_values[i]);
    }
  }

  {
    auto coo_mat = convert.to_coo(csc);
    EXPECT_EQ(coo_mat.nnz(), coo.nnz());
    auto row = coo_mat.outer_ptrs().view();
    auto col = coo_mat.inner_indices().view();
    auto values = coo_mat.values().view();
    for (int i = 0; i < coo_mat.nnz(); ++i) {
      EXPECT_EQ(row[i], h_coo_row_idx[i]);
      EXPECT_EQ(col[i], h_coo_col_idx[i]);
      EXPECT_FLOAT_EQ(values[i], h_coo_values[i]);
    }
  }
}

GTEST_TEST(coo, make_from_triplets) {
  std::vector<sparse::sparse_entry<float>> matrix_entry;
  matrix_entry.push_back({0, 0, 1.0f});
  matrix_entry.push_back({0, 1, 2.0f});
  matrix_entry.push_back({1, 1, 3.0f});
  matrix_entry.push_back({1, 2, 4.0f});
  matrix_entry.push_back({2, 2, 5.0f});

  auto coo = sparse::make_from_triplets<float>(matrix_entry.begin(), matrix_entry.end(), 3, 3);
  EXPECT_EQ(coo.nnz(), 5);
  auto row = coo.outer_ptrs().view();
  auto col = coo.inner_indices().view();
  auto values = coo.values().view();
  for (int i = 0; i < coo.nnz(); ++i) {
    EXPECT_EQ(row[i], matrix_entry[i].row_);
    EXPECT_EQ(col[i], matrix_entry[i].col_);
    EXPECT_FLOAT_EQ(values[i], matrix_entry[i].value_);
  }

  // duplicate entries
  matrix_entry.push_back({0, 0, 1.0f});
  matrix_entry.push_back({0, 1, 2.0f});

  auto coo2 = sparse::make_from_triplets<float>(matrix_entry.begin(), matrix_entry.end(), 3, 3);
  row = coo2.outer_ptrs().view();
  col = coo2.inner_indices().view();
  values = coo2.values().view();
  EXPECT_EQ(coo2.nnz(), 5);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(row[i], matrix_entry[i].row_);
    EXPECT_EQ(col[i], matrix_entry[i].col_);
    EXPECT_FLOAT_EQ(values[i], matrix_entry[i].value_ * 2);
  }

  for (int i = 2; i < coo2.nnz(); ++i) {
    EXPECT_EQ(row[i], matrix_entry[i].row_);
    EXPECT_EQ(col[i], matrix_entry[i].col_);
    EXPECT_FLOAT_EQ(values[i], matrix_entry[i].value_);
  }
}

GTEST_TEST(laplacian, 1d) {
  sparse::laplace_operator<float, 1> laplace(dshape<1>{4});

  auto matrix = laplace.matrix<sparse::sparse_format::csr>();
  auto mat = eigen_support::map(matrix.const_view());

  std::cout << mat.toDense() << std::endl;
}

GTEST_TEST(coo, io) {
  std::vector<sparse::sparse_entry<float>> matrix_entry;
  matrix_entry.push_back({0, 0, 1.0f});
  matrix_entry.push_back({0, 1, 2.0f});
  matrix_entry.push_back({1, 1, 3.0f});
  matrix_entry.push_back({1, 2, 4.0f});
  matrix_entry.push_back({2, 2, 5.0f});

  auto coo = sparse::make_from_triplets<float>(matrix_entry.begin(), matrix_entry.end(), 3, 3);

  io::matrix_market<float> io;
  {
    std::ofstream out("coo.mm");
    io.write(out, coo.const_view());
  }

  {
    std::ifstream in("coo.mm");
    auto coo2 = io.read(in);
    auto row = coo2.outer_ptrs().view();
    auto col = coo2.inner_indices().view();
    auto values = coo2.values().view();
    for (int i = 0; i < coo2.nnz(); ++i) {
      EXPECT_EQ(row[i], matrix_entry[i].row_);
      EXPECT_EQ(col[i], matrix_entry[i].col_);
      EXPECT_FLOAT_EQ(values[i], matrix_entry[i].value_);
    }
  }
}

GTEST_TEST(csr, spmm) {
  std::vector<sparse::sparse_entry<float>> matrix_entry;
  matrix_entry.push_back({0, 0, 1.0f});
  matrix_entry.push_back({0, 1, 2.0f});
  matrix_entry.push_back({1, 1, 3.0f});
  matrix_entry.push_back({1, 2, 4.0f});
  matrix_entry.push_back({2, 2, 5.0f});

  auto coo = sparse::make_from_triplets<float>(matrix_entry.begin(), matrix_entry.end(), 3, 3);
  auto csr = sparse::format_convert<float, sparse::sparse_format::csr, device::cpu>().from_coo(coo.const_view());

  auto eigen_csr = eigen_support::map(csr.const_view());

  auto x = Eigen::MatrixXf(4, 3);
  x.setRandom();
  auto y = Eigen::MatrixXf(4, 3);
  y.setZero();
  
  auto xv = eigen_support::view(x); // 3,4
  auto yv = eigen_support::view(y); // 3,4
  // mat: 3,3
  sparse::blas::eigen<float, sparse::sparse_format::csr> blas(csr.const_view());
  blas.spmm(1.0f, xv, 0.0f, yv);
  auto y_true = (eigen_csr * x.transpose()).transpose().eval();

  for (int i = 0; i < y.rows(); ++i) {
    for (int j = 0; j < y.cols(); ++j) {
      EXPECT_FLOAT_EQ(y(i, j), y_true(i, j));
    }
  }
}