#include <gtest/gtest.h>

#include "mathprim/geometry/laplacian.hpp"
#include "mathprim/geometry/lumped_mass.hpp"
#include "mathprim/supports/eigen_sparse.hpp"
#include "mathprim/parallel/parallel.hpp"

using namespace mathprim;

float tru_lap[4][4] = {
  {-0.5, 0, 0.5, 0},
  {0, -1, 1, 0},
  {0.5, 1, -2, 0.5},
  {0, 0, 0.5, -0.5}
};

GTEST_TEST(laplacian, manifold) {
  float vertices[] = {0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      1.0, 1.0, 0.0};
  index_t faces[] = {0, 1, 2,
                     1, 2, 3};
  auto vert = view(vertices, make_shape(4, 3));
  auto face = view(faces, make_shape(2, 3));
  geometry::basic_mesh<float, 3, 3, device::cpu> mesh{vert, face};
  auto lap = geometry::laplacian_builder<sparse::sparse_format::csr, float, 3, 3, device::cpu>::build(mesh);
  auto dense = eigen_support::map(lap.view()).toDense();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(dense(i, j), -tru_lap[i][j], 1e-6);
    }
  }
}

GTEST_TEST(laplacian, plane) {
  float vertices[] = {0.0, 0.0,
                      1.0, 0.0,
                      0.0, 1.0,
                      1.0, 1.0,};
  index_t faces[] = {0, 1, 2,
                     1, 2, 3};
  auto vert = view(vertices, make_shape(4, 2));
  auto face = view(faces, make_shape(2, 3));
  geometry::basic_mesh<float, 2, 3, device::cpu> mesh{vert, face};
  auto lap = geometry::laplacian_builder<sparse::sparse_format::csr, float, 2, 3, device::cpu>::build(mesh);
  auto dense = eigen_support::map(lap.view()).toDense();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(dense(i, j), -tru_lap[i][j], 1e-6);
    }
  }
}

GTEST_TEST(laplacian, tetrahedral) {
  // 2D cube:
  float vertices[] = {0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0,
                      1.0, 1.0, 0.0,
                      1.0, 0.0, 1.0,
                      0.0, 1.0, 1.0,
                      1.0, 1.0, 1.0};
  index_t faces[] = {0, 1, 2, 3,
                     1, 2, 4, 5,
                     2, 3, 6, 7,
                     3, 0, 5, 6,
                     0, 1, 4, 7,
                     4, 5, 6, 7};

  auto vert = view(vertices, make_shape(8, 3));
  auto face = view(faces, make_shape(6, 4));
  geometry::basic_mesh<float, 3, 4, device::cpu> mesh{vert, face};
  auto lap = geometry::build_laplacian<mathprim::sparse::sparse_format::csr>(mesh);
  auto dense = eigen_support::map(lap.view()).toDense();
  std::cout << dense << std::endl;
}

GTEST_TEST(mass, manifold) {
  float vertices[] = {0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      1.0, 1.0, 0.0};
  index_t faces[] = {0, 1, 2,
                     1, 2, 3};
  auto vert = view(vertices, make_shape(4, 3));
  auto face = view(faces, make_shape(2, 3));
  geometry::basic_mesh<float, 3, 3, device::cpu> mesh{vert, face};
  auto mass = geometry::lumped_mass_builder<sparse::sparse_format::csr, float, 3, 3, device::cpu>::build(mesh);
  auto dense = eigen_support::map(mass.view()).toDense();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        EXPECT_NEAR(dense(i, j), 0.25, 1e-6);
      } else {
        EXPECT_NEAR(dense(i, j), 0, 1e-6);
      }
    }
  }
}

GTEST_TEST(mass, 2d) {
  float vertices[] = {0.0, 0.0,
                      1.0, 0.0,
                      0.0, 1.0,
                      1.0, 1.0};
  index_t faces[] = {0, 1, 2,
                     1, 2, 3};
  auto vert = view(vertices, make_shape(4, 2));
  auto face = view(faces, make_shape(2, 3));
  geometry::basic_mesh<float, 2, 3, device::cpu> mesh{vert, face};
  auto mass = geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr>(mesh);
  auto dense = eigen_support::map(mass.view()).toDense();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        EXPECT_NEAR(dense(i, j), 0.25, 1e-6);
      } else {
        EXPECT_NEAR(dense(i, j), 0, 1e-6);
      }
    }
  }
}

GTEST_TEST(mass, tetrahedral) {
  // 2D cube:
  // 2D cube:
  float vertices[] = {0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0,
                      1.0, 1.0, 0.0,
                      1.0, 0.0, 1.0,
                      0.0, 1.0, 1.0,
                      1.0, 1.0, 1.0};
  index_t faces[] = {0, 1, 2, 3,
                     1, 2, 4, 5,
                     2, 3, 6, 7,
                     3, 0, 5, 6,
                     0, 1, 4, 7,
                     4, 5, 6, 7};

  auto vert = view(vertices, make_shape(8, 3));
  auto face = view(faces, make_shape(6, 4));
  geometry::basic_mesh<float, 3, 4, device::cpu> mesh{vert, face};
  auto mass = geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr>(mesh);
  auto dense = eigen_support::map(mass.view()).toDense();
  std::cout << dense << std::endl;
}