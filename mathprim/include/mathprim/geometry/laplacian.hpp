#pragma once
#include <Eigen/Geometry>  // cross

#include "mathprim/geometry/basic_mesh.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/sparse/cvt.hpp"
#include "mathprim/supports/eigen_dense.hpp"
namespace mathprim::geometry {
namespace internal {

template <typename Scalar, typename Point1, typename Point2, typename Point3>
MATHPRIM_PRIMFUNC Scalar cotangent(const Eigen::MatrixBase<Point1>& p1, const Eigen::MatrixBase<Point2>& p2,
                                   const Eigen::MatrixBase<Point3>& p3) {
  const auto v1 = (p2 - p1).eval();
  const auto v2 = (p3 - p1).eval();
  Scalar norm1 = v1.norm(), norm2 = v2.norm();
  Scalar cosine = v1.dot(v2) / (norm1 * norm2);
  Scalar sine = v1.cross(v2).norm() / (norm1 * norm2);
  return cosine / sine;
}

template <typename Scalar, typename Point1, typename Point2, typename Point3>
MATHPRIM_PRIMFUNC Scalar cotangent2d(const Eigen::MatrixBase<Point1>& p1, const Eigen::MatrixBase<Point2>& p2,
                                     const Eigen::MatrixBase<Point3>& p3) {
  const auto v1 = (p2 - p1).eval();
  const auto v2 = (p3 - p1).eval();
  Scalar norm1 = v1.norm(), norm2 = v2.norm();
  Scalar cosine = v1.dot(v2) / (norm1 * norm2);
  Scalar cross_norm = abs(v1.x() * v2.y() - v1.y() * v2.x());
  Scalar sine = cross_norm / (norm1 * norm2);
  return cosine / sine;
}
}  // namespace internal

template <sparse::sparse_format Format, typename Scalar, index_t SpaceNdim, index_t SimplexNdim, typename Device>
struct laplacian_builder;

// 2d on plane
template <sparse::sparse_format Format, typename Scalar>
struct laplacian_builder<Format, Scalar, 2, 3, device::cpu> {
  using mesh_t = basic_mesh<Scalar, 2, 3, device::cpu>;
  using sparse_t = sparse::basic_sparse_matrix<Scalar, device::cpu, Format>;
  using vertices = typename mesh_t::vertices;
  using indices = typename mesh_t::indices;
  using vector_view = contiguous_vector_view<const Scalar, device::cpu>;
  static sparse_t build(const mesh_t& mesh, vector_view elem_coeffs = {}) {
    using entry_t = sparse::entry<Scalar>;
    std::vector<entry_t> entries;
    if (elem_coeffs) {
      if (elem_coeffs.size() != mesh.indices_.shape(0)) {
        throw std::runtime_error("elem_coeffs must have the same size as the number of elements.");
      }
    }

    const index_t n_faces = mesh.indices_.shape(0);
    const index_t n_verts = mesh.vertices_.shape(0);
    entries.reserve(n_faces * 3 + n_verts);
    for (index_t i = 0; i < n_verts; ++i) {
      entries.push_back({i, i, 0});
    }

    for (index_t i = 0; i < n_faces; ++i) {
      auto face = mesh.indices_[i];
      for (index_t j = 0; j < 3; ++j) {
        const index_t v1 = face[j];
        const index_t v2 = face[(j + 1) % 3];
        const index_t v3 = face[(j + 2) % 3];
        auto p1 = eigen_support::cmap(mesh.vertices_[v1]);
        auto p2 = eigen_support::cmap(mesh.vertices_[v2]);
        auto p3 = eigen_support::cmap(mesh.vertices_[v3]);
        Scalar val = internal::cotangent2d<Scalar>(p1, p2, p3) * static_cast<Scalar>(0.5);
        if (elem_coeffs) {
          val *= elem_coeffs[i];
        }
        // Off-diagonal entries
        entries.push_back({v1, v2, -val});
        entries.push_back({v2, v1, -val});
        // Diagonal entries
        entries[v1].value_ += val;
        entries[v2].value_ += val;
      }
    }
    auto coo = sparse::make_from_triplets<Scalar>(  //
        entries.begin(), entries.end(),             // triplets
        n_verts, n_verts);                          // shape
    if constexpr (Format == sparse::sparse_format::coo) {
      return coo;
    } else {
      return sparse::make_from_coos<Scalar, Format>(coo);
    }
  }
};

// 2d manifold in 3d
template <sparse::sparse_format Format, typename Scalar>
struct laplacian_builder<Format, Scalar, 3, 3, device::cpu> {
  using mesh_t = basic_mesh<Scalar, 3, 3, device::cpu>;
  using sparse_t = sparse::basic_sparse_matrix<Scalar, device::cpu, Format>;
  using vertices = typename mesh_t::vertices;
  using indices = typename mesh_t::indices;
  using vector_view = contiguous_vector_view<const Scalar, device::cpu>;
  static sparse_t build(const mesh_t& mesh, vector_view elem_coeffs = {}) {
    using entry_t = sparse::entry<Scalar>;
    std::vector<entry_t> entries;
    if (elem_coeffs) {
      if (elem_coeffs.size() != mesh.indices_.shape(0)) {
        throw std::runtime_error("elem_coeffs must have the same size as the number of elements.");
      }
    }
    const index_t n_faces = mesh.indices_.shape(0);
    const index_t n_verts = mesh.vertices_.shape(0);
    entries.reserve(n_faces * 3 + n_verts);
    for (index_t i = 0; i < n_verts; ++i) {
      entries.push_back({i, i, 0});
    }

    for (index_t i = 0; i < n_faces; ++i) {
      auto face = mesh.indices_[i];
      for (index_t j = 0; j < 3; ++j) {
        const index_t v1 = face[j];
        const index_t v2 = face[(j + 1) % 3];
        const index_t v3 = face[(j + 2) % 3];
        auto p1 = eigen_support::cmap(mesh.vertices_[v1]);
        auto p2 = eigen_support::cmap(mesh.vertices_[v2]);
        auto p3 = eigen_support::cmap(mesh.vertices_[v3]);
        Scalar val = internal::cotangent<Scalar>(p1, p2, p3) * static_cast<Scalar>(0.5);
        if (elem_coeffs) {
          val *= elem_coeffs[i];
        }
        // Off-diagonal entries
        entries.push_back({v1, v2, -val});
        entries.push_back({v2, v1, -val});
        // Diagonal entries
        entries[v1].value_ += val;
        entries[v2].value_ += val;
      }
    }
    auto coo = sparse::make_from_triplets<Scalar>(  //
        entries.begin(), entries.end(),             // triplets
        n_verts, n_verts);                          // shape
    if constexpr (Format == sparse::sparse_format::coo) {
      return coo;
    } else {
      return sparse::make_from_coos<Scalar, Format>(coo);
    }
  }
};

// 3D tetrahedral mesh
template <sparse::sparse_format Format, typename Scalar>
struct laplacian_builder<Format, Scalar, 3, 4, device::cpu> {
  using mesh_t = basic_mesh<Scalar, 3, 4, device::cpu>;
  using sparse_t = sparse::basic_sparse_matrix<Scalar, device::cpu, Format>;
  using vertices = typename mesh_t::vertices;
  using indices = typename mesh_t::indices;
  using entry = sparse::entry<Scalar>;
  using vector_view = contiguous_vector_view<const Scalar, device::cpu>;

  static sparse_t build(const mesh_t& mesh, vector_view elem_coeffs = {}) {
    using entry_t = sparse::entry<Scalar>;
    std::vector<entry_t> entries;
    const index_t n_elems = mesh.indices_.shape(0);
    const index_t n_verts = mesh.vertices_.shape(0);
    entries.reserve(4 * 4 * n_elems);
    for (index_t i = 0; i < n_elems; ++i) {
      auto elem = mesh.indices_[i];
      Eigen::Matrix<Scalar, 4, 4> c;
      for (index_t j = 0; j < 4; ++j) {
        for (index_t k = 0; k < 3; ++k) {
          c(j, k) = mesh.vertices_(elem[j], k);
        }
        c(j, 3) = 1;
      }
      Scalar volume = std::abs(c.determinant()) / 6;
      c = c.inverse().eval();
      Eigen::Matrix<Scalar, 4, 4> local;
      for (index_t i = 0; i <= 3; ++i) {
        for (index_t j = 0; j <= 3; ++j) {
          Scalar lap = 0.;
          for (index_t d = 0; d < 3; ++d) {
            lap += c(d, i) * c(d, j);
          }
          local(i, j) = lap * volume;
        }
      }
      if (elem_coeffs) {
        local *= elem_coeffs[i];
      }

      // TODO: Make spsd enough.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 4, 4>> eigensolver(local);
      if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen solver failed.");
      }
      Eigen::Matrix<Scalar, 4, 4> eigenvectors = eigensolver.eigenvectors();
      Eigen::Vector<Scalar, 4> eigvals = eigensolver.eigenvalues();
      local = eigenvectors * eigvals.asDiagonal() * eigenvectors.transpose();

      for (index_t i = 0; i < 4; ++i) {
        for (index_t j = 0; j < 4; ++j) {
          entries.push_back({elem[i], elem[j], local(i, j)});
        }
      }
    }
    auto coo = sparse::make_from_triplets<Scalar>(  //
        entries.begin(), entries.end(),             // triplets
        n_verts, n_verts);                          // shape
    if constexpr (Format == sparse::sparse_format::coo) {
      return coo;
    } else {
      return sparse::make_from_coos<Scalar, Format>(coo);
    }
  }
};

template <sparse::sparse_format Format, typename Scalar, index_t SpaceNdim, index_t SimplexNdim, typename Device>
sparse::basic_sparse_matrix<Scalar, Device, Format> build_laplacian(
    const basic_mesh<Scalar, SpaceNdim, SimplexNdim, Device>& mesh) {
  return laplacian_builder<Format, Scalar, SpaceNdim, SimplexNdim, Device>::build(mesh);
}

template <sparse::sparse_format Format, typename Scalar, index_t SpaceNdim, index_t SimplexNdim, typename Device>
sparse::basic_sparse_matrix<Scalar, Device, Format> build_laplacian(
    const basic_mesh<Scalar, SpaceNdim, SimplexNdim, Device>& mesh,
    contiguous_vector_view<const Scalar, Device> elem_coeffs) {
  return laplacian_builder<Format, Scalar, SpaceNdim, SimplexNdim, Device>::build(mesh, elem_coeffs);
}

}  // namespace mathprim::geometry