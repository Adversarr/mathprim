#pragma once

#include <Eigen/Geometry>  // cross

#include "mathprim/geometry/basic_mesh.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/sparse/cvt.hpp"
#include "mathprim/supports/eigen_dense.hpp"
namespace mathprim::geometry {

template <sparse::sparse_format Format, typename Scalar, index_t SpaceNdim, index_t SimplexNdim, typename Device>
struct lumped_mass_builder;



/* === 2d on plane ===
   Equivalent Python code: 
   Transformed from https://github.com/libigl/libigl/blob/main/include/igl/massmatrix_intrinsic.cpp

```py
def mass_matrix_voronoi(vertices, faces):
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    M = np.zeros((num_vertices, num_vertices))

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Compute edge lengths
        l0 = np.linalg.norm(v1 - v2)
        l1 = np.linalg.norm(v2 - v0)
        l2 = np.linalg.norm(v0 - v1)

        # Compute cosines
        cos0 = (l1**2 + l2**2 - l0**2) / (2 * l1 * l2)
        cos1 = (l0**2 + l2**2 - l1**2) / (2 * l0 * l2)
        cos2 = (l0**2 + l1**2 - l2**2) / (2 * l0 * l1)

        # Compute Voronoi weights
        quads = np.zeros(3)
        if cos0 < 0:
            quads[0] = 0.25 * area
            quads[1] = 0.125 * area
            quads[2] = 0.125 * area
        elif cos1 < 0:
            quads[0] = 0.125 * area
            quads[1] = 0.25 * area
            quads[2] = 0.125 * area
        elif cos2 < 0:
            quads[0] = 0.125 * area
            quads[1] = 0.125 * area
            quads[2] = 0.25 * area
        else:
            quads[0] = (cos1 * l1**2 + cos2 * l2**2) / 8
            quads[1] = (cos2 * l2**2 + cos0 * l0**2) / 8
            quads[2] = (cos0 * l0**2 + cos1 * l1**2) / 8

        # Update mass matrix
        M[face[0], face[0]] += quads[0]
        M[face[1], face[1]] += quads[1]
        M[face[2], face[2]] += quads[2]

    return csr_matrix(M)
```
*/
namespace internal {
template <typename Scalar>
Eigen::Vector3<Scalar> local_mass(
  Eigen::Vector3<Scalar> v0, Eigen::Vector3<Scalar> v1, Eigen::Vector3<Scalar> v2) {
  Eigen::Vector3<Scalar> mass;
  const Scalar area = (v1 - v0).cross(v2 - v0).norm();
  const Scalar l0 = (v1 - v2).norm();
  const Scalar l1 = (v2 - v0).norm();
  const Scalar l2 = (v0 - v1).norm();
  const Scalar cos0 = (l1 * l1 + l2 * l2 - l0 * l0) / (2 * l1 * l2);
  const Scalar cos1 = (l0 * l0 + l2 * l2 - l1 * l1) / (2 * l0 * l2);
  const Scalar cos2 = (l0 * l0 + l1 * l1 - l2 * l2) / (2 * l0 * l1);
  const Scalar eps = std::numeric_limits<Scalar>::epsilon();
  if (cos0 <= eps) {
    // mass << 0.25 * area, 0.125 * area, 0.125 * area;
    mass[0] = static_cast<Scalar>(0.25) * area;
    mass[1] = static_cast<Scalar>(0.125) * area;
    mass[2] = static_cast<Scalar>(0.125) * area;
  } else if (cos1 <= eps) {
    // mass << 0.125 * area, 0.25 * area, 0.125 * area;
    mass[0] = static_cast<Scalar>(0.125) * area;
    mass[1] = static_cast<Scalar>(0.25) * area;
    mass[2] = static_cast<Scalar>(0.125) * area;
  } else if (cos2 <= eps) {
    // mass << 0.125 * area, 0.125 * area, 0.25 * area;
    // mass = {0.125 * area, 0.125 * area, 0.25 * area};
    mass[0] = static_cast<Scalar>(0.125) * area;
    mass[1] = static_cast<Scalar>(0.125) * area;
    mass[2] = static_cast<Scalar>(0.25) * area;
  } else {
    // mass << (cos1 * l1 * l1 + cos2 * l2 * l2) / 8,
    //         (cos2 * l2 * l2 + cos0 * l0 * l0) / 8,
    //         (cos0 * l0 * l0 + cos1 * l1 * l1) / 8;
    mass[0] = (cos1 * l1 * l1 + cos2 * l2 * l2) / 8;
    mass[1] = (cos2 * l2 * l2 + cos0 * l0 * l0) / 8;
    mass[2] = (cos0 * l0 * l0 + cos1 * l1 * l1) / 8;
  }
  return mass;
}
}  // namespace internal

template <sparse::sparse_format Format, typename Scalar>
struct lumped_mass_builder<Format, Scalar, 3, 3, device::cpu> {
  using mesh_t = basic_mesh<Scalar, 3, 3, device::cpu>;
  using sparse_t = sparse::basic_sparse_matrix<Scalar, device::cpu, Format>;
  using vertices = typename mesh_t::vertices;
  using indices = typename mesh_t::indices;
  static sparse_t build(const mesh_t& mesh) {
    using entry_t = sparse::sparse_entry<Scalar>;
    std::vector<entry_t> entries;
    const index_t n_faces = mesh.indices_.shape(0);
    const index_t n_verts = mesh.vertices_.shape(0);
    entries.reserve(n_faces * 3 + n_verts);
    for (index_t i = 0; i < n_verts; ++i) {
      entries.push_back({i, i, 0});
    }

    for (index_t i = 0; i < n_faces; ++i) {
      auto face = mesh.indices_[i];
      const auto p1 = eigen_support::cmap(mesh.vertices_[face[0]]);
      const auto p2 = eigen_support::cmap(mesh.vertices_[face[1]]);
      const auto p3 = eigen_support::cmap(mesh.vertices_[face[2]]);
      const auto mass = internal::local_mass<Scalar>(p1, p2, p3);
      entries[face[0]].value_ += mass[0];
      entries[face[1]].value_ += mass[1];
      entries[face[2]].value_ += mass[2];
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

template <sparse::sparse_format Format, typename Scalar>
struct lumped_mass_builder<Format, Scalar, 2, 3, device::cpu> {
  using mesh_t = basic_mesh<Scalar, 2, 3, device::cpu>;
  using sparse_t = sparse::basic_sparse_matrix<Scalar, device::cpu, Format>;
  using vertices = typename mesh_t::vertices;
  using indices = typename mesh_t::indices;
  static sparse_t build(const mesh_t& mesh) {
    using entry_t = sparse::sparse_entry<Scalar>;
    std::vector<entry_t> entries;
    const index_t n_faces = mesh.indices_.shape(0);
    const index_t n_verts = mesh.vertices_.shape(0);
    entries.reserve(n_faces * 3 + n_verts);
    for (index_t i = 0; i < n_verts; ++i) {
      entries.push_back({i, i, 0});
    }

    for (index_t i = 0; i < n_faces; ++i) {
      auto face = mesh.indices_[i];
      const auto p1 = eigen_support::cmap(mesh.vertices_[face[0]]);
      const auto p2 = eigen_support::cmap(mesh.vertices_[face[1]]);
      const auto p3 = eigen_support::cmap(mesh.vertices_[face[2]]);
      const auto mass = internal::local_mass<Scalar>({p1[0], p1[1], 0}, {p2[0], p2[1], 0}, {p3[0], p3[1], 0});
      entries[face[0]].value_ += mass[0];
      entries[face[1]].value_ += mass[1];
      entries[face[2]].value_ += mass[2];
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

// For Tetras, just volumn/4
template <sparse::sparse_format Format, typename Scalar>
struct lumped_mass_builder<Format, Scalar, 3, 4, device::cpu> {
  using mesh_t = basic_mesh<Scalar, 3, 4, device::cpu>;
  using sparse_t = sparse::basic_sparse_matrix<Scalar, device::cpu, Format>;
  using vertices = typename mesh_t::vertices;
  using indices = typename mesh_t::indices;
  static sparse_t build(const mesh_t& mesh) {
    using entry_t = sparse::sparse_entry<Scalar>;
    std::vector<entry_t> entries;
    const index_t n_elems = mesh.indices_.shape(0);
    const index_t n_verts = mesh.vertices_.shape(0);
    entries.reserve(n_elems * 3 + n_verts);
    for (index_t i = 0; i < n_verts; ++i) {
      entries.push_back({i, i, 0});
    }

    for (index_t i = 0; i < n_elems; ++i) {
      auto face = mesh.indices_[i];
      const auto p1 = eigen_support::cmap(mesh.vertices_[face[0]]);
      const auto p2 = eigen_support::cmap(mesh.vertices_[face[1]]);
      const auto p3 = eigen_support::cmap(mesh.vertices_[face[2]]);
      const auto p4 = eigen_support::cmap(mesh.vertices_[face[3]]);
      const auto mass = ::abs((p2 - p1).cross(p3 - p1).dot(p4 - p1) / 6);

      entries[face[0]].value_ += mass / 4;
      entries[face[1]].value_ += mass / 4;
      entries[face[2]].value_ += mass / 4;
      entries[face[3]].value_ += mass / 4;
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
sparse::basic_sparse_matrix<Scalar, Device, Format> build_lumped_mass(
    const basic_mesh<Scalar, SpaceNdim, SimplexNdim, Device>& mesh) {
  return lumped_mass_builder<Format, Scalar, SpaceNdim, SimplexNdim, Device>::build(mesh);
}

}  // namespace mathprim::geometry