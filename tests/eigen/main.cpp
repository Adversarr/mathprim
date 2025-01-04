#include <iostream>
#include <mathprim/core/buffer.hpp>
#include <mathprim/core/buffer_view.hpp>
#include <mathprim/supports/eigen_dense.hpp>
using namespace mathprim;

int main() {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  auto matrix = make_buffer<float>(4, 4);
  auto mv = matrix.view();

  for (auto [i, j] : mv.shape()) {
    mv(i, j) = i * 4 + j;
  }

  auto map_to_matrix = eigen_support::cmap<4, 4>(mv.as_const());
  std::cout << map_to_matrix << std::endl;

  auto vector = make_buffer<float>(4);
  auto vv = vector.view();
  auto map_to_vector = eigen_support::cmap<4>(vv.as_const());

  for (auto [i] : vv.shape()) {
    vv(i) = i;
  }

  std::cout << map_to_vector << std::endl;

  // Another way, use map, ignores the continuous property.
  auto map_to_matrix2 = eigen_support::map<4, 4>(mv.as_const());
  std::cout << map_to_matrix2 << std::endl;
  auto map_to_vector2 = eigen_support::map<4>(vv.as_const());
  std::cout << map_to_vector2 << std::endl;

  // this should assert because the shape is not the same.
  // auto map_to_matrix3 = eigen_support::map<3, 4>(mv.as_const());
  return 0;
}