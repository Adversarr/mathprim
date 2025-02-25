#include <iostream>

#include "mathprim/linalg/direct/cholmod.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/blas/naive.hpp"
#include "mathprim/sparse/systems/laplace.hpp"
#include "mathprim/supports/eigen_sparse.hpp"
#include "mathprim/supports/stringify.hpp"
using namespace mathprim;

int main() {
  int dsize = 4;
  sparse::laplace_operator<float, 2> lap(make_shape(dsize, dsize));
  auto mat_buf = lap.matrix<mathprim::sparse::sparse_format::csr>();
  auto mat = mat_buf.const_view();
  auto rows = mat.rows();
  auto b = make_buffer<float>(rows);
  auto x = make_buffer<float>(rows);

  sparse::blas::naive<float, sparse::sparse_format::csr, par::seq> bl{mat};
  // GT = ones.
  par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
    xv[i] = 1.0f;
  });
  // b = A * x
  bl.gemv(1.0f, x.view(), 0.0f, b.view());

  // x = (i % 100 - 50) / 100.0f
  par::seq().run(make_shape(rows), [xv = x.view()](index_t i) {
    xv[i] = (i % 100 - 50) / 100.0f;
  });

  // solve
  auto chol = sparse::direct::cholmod_chol<float, device::cpu>{mat};
  x.fill_bytes(0);
  chol.solve(x.view(), b.view()); // A x = b
  // checking
  auto b2 = make_buffer<float>(rows);
  bl.gemv(1.0f, x.view(), 0.0f, b2.view());
  auto bv = b.view(), b2v = b2.view();
  for (index_t i = 0; i < rows; ++i) {
    if (std::abs(bv[i] - b2v[i]) > 1e-6f) {
      std::cerr << "Error at " << i << " " << bv[i] << " " << b2v[i] << std::endl;
    } else {
      
    }
  }
  std::cout << "Correct" << std::endl;
  return 0;
}