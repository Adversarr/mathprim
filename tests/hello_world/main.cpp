#include <iostream>
#include <iterator>
#include <mathprim/core/buffer.hpp>
#include <mathprim/supports/eigen_dense.hpp>
#include <mathprim/supports/eigen_sparse.hpp>
#include <mathprim/supports/view_from/stl.hpp>
using namespace mathprim;
using namespace mathprim::literal;

int main() {
  float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto v = view(buf, shape_t<4>());
  // auto v2 = view(buf);
  // const view:
  basic_view<const float, shape_t<4>, stride_t<1>, device::cpu> v2 = v;
  // Should fail:
  // basic_view<const float, shape_t<4>, stride_t<2>, device::cpu> v3 = v;
  // basic_view<const float, shape_t<2>, stride_t<4>, device::cpu> v3 = v;
  basic_view<const float, shape_t<-1>, stride_t<1>, device::cpu> dyn_1 = v;
  // Should fail:
  // basic_view<float, shape_t<-1>, stride_t<4>, device::cpu> non_const = dyn_1;

  using T = decltype(v.begin());

  int n = 4;
  make_shape(41_s, n);

  Eigen::SparseMatrix<float> mat(5, 4);
  mat.insert(1, 2) = 1.0f;
  mat.insert(2, 3) = 2.0f;
  mat.insert(3, 1) = 3.0f;
  mat.insert(4, 0) = 4.0f;
  mat.makeCompressed();

  auto a_buf = make_buffer<float>(4, 3);
  auto b_buf = make_buffer<float>(5, 3);
  for (auto [i, j]: make_shape(4, 3)) {
    a_buf.view()(i, j) = i * 3 + j;
  }
  for (auto [i, j]: make_shape(5, 3)) {
    b_buf.view()(i, j) = i * 5 + j;
  }

  auto a = eigen_support::map(a_buf.view()); // 3, 4
  auto b = eigen_support::map(b_buf.view()); // 3, 5
  std::cout << a << std::endl;
  std::cout << b << std::endl;

  // 4, 3 = 4, 5 * 5, 3
  a.transpose() = mat.transpose() * b.transpose();
  std::cout << a.transpose() << std::endl;
}
