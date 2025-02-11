#include "mathprim/parallel/openmp.hpp"
#include <mathprim/sparse/blas/csr.hpp>
#include <gtest/gtest.h>

using namespace mathprim;
GTEST_TEST(csr, gemv) {
  const int rows = 3, cols = 3, nnz = 5;
  float h_csr_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int h_csr_col_idx[] = {0, 1, 1, 2, 0};
  int h_csr_row_ptr[] = {0, 2, 4, 5};

  sparse::basic_sparse_view<float, device::cpu, sparse::sparse_format::csr, true> mat(
    view(h_csr_values, make_shape(nnz)).as_const(),
    view(h_csr_row_ptr, make_shape(rows + 1)).as_const(),
    view(h_csr_col_idx, make_shape(nnz)).as_const(),
    rows,
    cols,
    nnz,
    sparse::sparse_property::general,
    false
  );

  float h_x[] = {1.0f, 1.0f, 1.0f};
  float h_y[rows] = {0.0f};
  auto x = view(h_x, make_shape(cols));
  auto y = view(h_y, make_shape(rows));

  {
    sparse::blas::naive<float, device::cpu, sparse::sparse_format::csr> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y);

    // y[0] = 1*1 + 2*1 = 3.00
    // y[1] = 3*1 + 4*1 = 7.00
    // y[2] = 5*1 = 5.00
    EXPECT_FLOAT_EQ(h_y[0], 3.0f);
    EXPECT_FLOAT_EQ(h_y[1], 7.0f);
    EXPECT_FLOAT_EQ(h_y[2], 5.0f);
  }

  {
    sparse::blas::naive<float, device::cpu, sparse::sparse_format::csr, par::openmp> blas(mat);
    blas.gemv(1.0f, x, 0.0f, y);

    // y[0] = 1*1 + 2*1 = 3.00
    // y[1] = 3*1 + 4*1 = 7.00
    // y[2] = 5*1 = 5.00
    EXPECT_FLOAT_EQ(h_y[0], 3.0f);
    EXPECT_FLOAT_EQ(h_y[1], 7.0f);
    EXPECT_FLOAT_EQ(h_y[2], 5.0f);
  }
}