#include <gtest/gtest.h>

#include "mathprim/blas/cpu_blas.hpp"
#include "mathprim/dnn/nn/activation.hpp"
#include "mathprim/dnn/nn/global_sparse.hpp"
#include "mathprim/dnn/nn/sequential.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/blas/eigen.hpp"
#include "mathprim/sparse/systems/laplace.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;

constexpr index_t B = 5; // batch_size

constexpr auto COMP = sparse::sparse_format::csr;

GTEST_TEST(global_sparse, self) {
  using sparse_t = sparse::blas::eigen<float, COMP>;
  using sparse_nn = dnn::global_sparse<sparse_t>;
  auto laplacian = sparse::laplace_operator<float, 1>(make_shape(10)).matrix<COMP>();

  sparse_nn gs(sparse_t(laplacian.const_view()), 3, false);
  auto ctx = dnn::make_ctx(gs, blas::cpu_blas<float>{}, par::seq{});
  ctx.compile(gs, B);

  auto x = ctx.input(); // [B, 10, 3]
  for (index_t b = 0; b < B; ++b) {
    for (index_t i = 0; i < 10; ++i) {
      for (index_t j = 0; j < 3; ++j) {
        x(b, i, j) = i + j;
      }
    }
  }

  auto y_true_buf = make_buffer<float>(B, 10, 3);
  auto y_true = y_true_buf.view();
  ctx.forward(gs);

  for (index_t i = 0; i < B; ++i) {
    auto xi = eigen_support::cmap(x[i]); // 3, 10
    auto yi = eigen_support::cmap(y_true[i]); // 3, 10
    yi = xi * eigen_support::map(laplacian.const_view());
  }

  auto y = gs.output();
  for (auto [b, i, j]: y.shape()) {
    EXPECT_NEAR(y(b, i, j), y_true(b, i, j), 1e-4);
  }
}
