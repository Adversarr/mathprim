#include "geometry.hpp"

#include <Eigen/Sparse>
#include <iostream>

#include "mathprim/geometry/laplacian.hpp"
#include "mathprim/geometry/lumped_mass.hpp"
#include "mathprim/supports/eigen_sparse.hpp"

template <typename Flt>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> laplacian(
    nb::ndarray<Flt, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> vertices,
    nb::ndarray<int, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> faces) {
  if (vertices.ndim() != 2) {
    throw std::runtime_error("vertices must be 2D array");
  }
  if (faces.ndim() != 2) {
    throw std::runtime_error("faces must be 2D array");
  }

  auto nvert = static_cast<mp::index_t>(vertices.shape(0));
  auto nface = static_cast<mp::index_t>(faces.shape(0));
  auto ndim = static_cast<mp::index_t>(vertices.shape(1));
  auto dsimplex = static_cast<mp::index_t>(faces.shape(1));

  using namespace mp::literal;
  if (ndim == 2 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 2, 3, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 2_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 3, 3, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 4) {
    mp::geometry::basic_mesh<Flt, 3, 4, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 4_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else {
    std::ostringstream oss;
    oss << "Unsupported dimension or simplex type: ndim=" << ndim << ", dsimplex=" << dsimplex;
    throw std::runtime_error(oss.str());
  }
}

template <typename Flt>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> weighted_laplacian(
    nb::ndarray<Flt, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> vertices,
    nb::ndarray<int, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig> faces,
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> elem_weights) {
  if (vertices.ndim() != 2) {
    throw std::runtime_error("vertices must be 2D array");
  }
  if (faces.ndim() != 2) {
    throw std::runtime_error("faces must be 2D array");
  }

  auto nvert = static_cast<mp::index_t>(vertices.shape(0));
  auto nface = static_cast<mp::index_t>(faces.shape(0));
  auto ndim = static_cast<mp::index_t>(vertices.shape(1));
  auto dsimplex = static_cast<mp::index_t>(faces.shape(1));
  auto nweights = static_cast<mp::index_t>(elem_weights.shape(0));
  auto weights = nbex::to_mp_view_standard(elem_weights).as_const();

  if (nweights != nface) {
    throw std::runtime_error("elem_weights must have the same length as the number of faces");
  }


  using namespace mp::literal;
  if (ndim == 2 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 2, 3, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 2_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh, weights);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 3, 3, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh, weights);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 4) {
    mp::geometry::basic_mesh<Flt, 3, 4, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 4_s))};
    auto matrix = mp::geometry::build_laplacian<mathprim::sparse::sparse_format::csr, Flt>(mesh, weights);
    return mp::eigen_support::map(matrix.view());
  } else {
    std::ostringstream oss;
    oss << "Unsupported dimension or simplex type: ndim=" << ndim << ", dsimplex=" << dsimplex;
    throw std::runtime_error(oss.str());
  }
}

template <typename Flt>
Eigen::SparseMatrix<Flt, Eigen::RowMajor> lumped_mass(
    nb::ndarray<Flt, nb::shape<-1, -1>, nb::device::cpu, nb::numpy> vertices,
    nb::ndarray<int, nb::shape<-1, -1>, nb::device::cpu, nb::numpy> faces) {
  if (vertices.ndim() != 2) {
    throw std::runtime_error("vertices must be 2D array");
  }
  if (faces.ndim() != 2) {
    throw std::runtime_error("faces must be 2D array");
  }

  auto nvert = static_cast<mp::index_t>(vertices.shape(0));
  auto nface = static_cast<mp::index_t>(faces.shape(0));
  auto ndim = static_cast<mp::index_t>(vertices.shape(1));
  auto dsimplex = static_cast<mp::index_t>(faces.shape(1));

  using namespace mp::literal;
  if (ndim == 2 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 2, 3, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 2_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 3) {
    mp::geometry::basic_mesh<Flt, 3, 3, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 3_s))};
    auto matrix = mp::geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else if (ndim == 3 && dsimplex == 4) {
    mp::geometry::basic_mesh<Flt, 3, 4, mp::device::cpu> mesh{mp::view(vertices.data(), mp::make_shape(nvert, 3_s)),
                                                              mp::view(faces.data(), mp::make_shape(nface, 4_s))};
    auto matrix = mp::geometry::build_lumped_mass<mathprim::sparse::sparse_format::csr, Flt>(mesh);
    return mp::eigen_support::map(matrix.view());
  } else {
    std::ostringstream oss;
    oss << "Unsupported dimension or simplex type: ndim=" << ndim << ", dsimplex=" << dsimplex;
    throw std::runtime_error(oss.str());
  }
}

void bind_geometry(nb::module_& m) {
  m                                                    //
      .def("laplacian", laplacian<float>,              //
           "Compute the Laplacian matrix of a mesh.",  //
           nb::arg("vertices").noconvert(), nb::arg("faces").noconvert())
      .def("laplacian", laplacian<double>,             //
           "Compute the Laplacian matrix of a mesh.",  //
           nb::arg("vertices").noconvert(), nb::arg("faces").noconvert())
      .def("weighted_laplacian", weighted_laplacian<float>,     //
           "Compute the Weighted Laplacian matrix of a mesh.",  //
           nb::arg("vertices").noconvert(), nb::arg("faces").noconvert(), nb::arg("elem_weights").noconvert())
      .def("weighted_laplacian", weighted_laplacian<double>,    //
           "Compute the Weighted Laplacian matrix of a mesh.",  //
           nb::arg("vertices").noconvert(), nb::arg("faces").noconvert(), nb::arg("elem_weights").noconvert())
      .def("lumped_mass", lumped_mass<float>,            //
           "Compute the Lumped-mass matrix of a mesh.",  //
           nb::arg("vertices").noconvert(), nb::arg("faces").noconvert())
      .def("lumped_mass", lumped_mass<double>,           //
           "Compute the Lumped-mass matrix of a mesh.",  //
           nb::arg("vertices").noconvert(), nb::arg("faces").noconvert());
}
