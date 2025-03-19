
#include <mathprim/blas/cpu_handmade.hpp>
#include <mathprim/core/buffer.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/sparse/blas/eigen.hpp>
#include <mathprim/sparse/blas/naive.hpp>

using namespace mathprim;

int dsize = 100;

int main() {
  // 1 D laplacian
  const int rows = dsize, cols = dsize, nnz = dsize * 3 - 2;

  auto h_csr_values = make_buffer<float>(nnz);
  auto h_csr_col_idx = make_buffer<index_t>(nnz);
  auto h_csr_row_ptr = make_buffer<index_t>(rows + 1);
  auto values = h_csr_values.view();
  auto col_idx = h_csr_col_idx.view();
  auto row_ptr = h_csr_row_ptr.view();

  for (int i = 0; i < rows; i++) {
    if (i == 0) {
      row_ptr[i] = 0;
      col_idx[i * 3] = 0;
      values[i * 3] = 2.0f;
      col_idx[i * 3 + 1] = 1;
      values[i * 3 + 1] = -1.0f;
    } else if (i == rows - 1) {
      row_ptr[i] = nnz - 2;
      col_idx[nnz - 2] = i - 1;
      values[nnz - 2] = -1.0f;
      col_idx[nnz - 1] = i;
      values[nnz - 1] = 2.0f;
    } else {
      row_ptr[i] = i * 3 - 1;
      col_idx[i * 3 - 1] = i - 1;
      values[i * 3 - 1] = -1.0f;
      col_idx[i * 3] = i;
      values[i * 3] = 2.0f;
      col_idx[i * 3 + 1] = i + 1;
      values[i * 3 + 1] = -1.0f;
    }
  }
  row_ptr[rows] = nnz;

  sparse::basic_sparse_view<const float, device::cpu, sparse::sparse_format::csr> mat(
      values.as_const(), row_ptr.as_const(), col_idx.as_const(), rows, cols, nnz, sparse::sparse_property::general);

  using linear_op = sparse::blas::eigen<float, sparse::sparse_format::csr>;
  sparse::iterative::cg<float, device::cpu, linear_op, blas::cpu_handmade<float>> cg{mat};

  auto b = make_buffer<float>(rows);
  auto x = make_buffer<float>(rows);
  // GT = ones.
  par::seq().run(rows, [xv = x.view()](index_t i) {
    xv[i] = 1.0f;
  });
  // b = A * x
  cg.linear_operator().gemv(1.0f, x.view(), 0.0f, b.view());

  par::seq().run(rows, [xv = x.view(), bv = b.view()](index_t i) {
    xv[i] = (rand() % 100) / 100.0f;
  });

  cg.solve(x.view(), b.view());
  return 0;
}