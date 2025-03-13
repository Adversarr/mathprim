#pragma once
#include <vector>

#include "mathprim/sparse/cvt.hpp"

namespace mathprim::sparse {

template <typename Scalar, index_t Dim>
class laplace_operator {
public:
  using dimension_type = dshape<Dim>;
  explicit laplace_operator(const dimension_type &dims) : dims_(dims) {}

  struct bc_dirichlet {
    inline void operator()(const index_array<Dim> & /* self_idx */,      //
                           const index_array<Dim> & /* neighbor_idx */,  //
                           Scalar & /* self_value */,                    //
                           Scalar & /* neighbor_value */) const noexcept {}
  };

  /**
   * @brief Adopt the OpenVDB like API. the Fn should be a callable object that operates on
   *            (const index_array<Dim> &self_idx,
   *             const index_array<Dim> &neighbor_idx,
   *             Scalar &self_value,
   *             Scalar &neighbor_value) -> void
   *        Modify the self_value/neighbor_value to set the matrix for boundaries.
   *
   * @tparam SparseCompression
   * @tparam Fn
   * @param bc
   * @return basic_sparse_matrix<Scalar, device::cpu, SparseCompression>
   */
  template <sparse_format SparseCompression, typename Fn = bc_dirichlet>
  basic_sparse_matrix<Scalar, device::cpu, SparseCompression> matrix(Fn bc = bc_dirichlet{}) const;

  dimension_type dims() const noexcept {
    return dims_;
  }

private:
  dimension_type dims_;
};

template <typename Scalar, index_t Dim>
template <sparse_format SparseCompression, typename Fn>
basic_sparse_matrix<Scalar, device::cpu, SparseCompression> laplace_operator<Scalar, Dim>::matrix(Fn bc) const {
  std::vector<entry<Scalar>> entries;
  auto linear_strides = make_default_stride<Scalar>(dims_);
  for (auto cell_idx : dims_) {
    // For each neighbor of cell_idx
    Scalar self_value = static_cast<Scalar>(2 * Dim);
    auto self_idx_linear = sub2ind(linear_strides, cell_idx);
    for (index_t i = 0; i < Dim; ++i) {
      index_array<Dim> neighbor_idx = cell_idx;
      neighbor_idx[i] += 1;
      if (is_in_bound(dims_, neighbor_idx)) {
        Scalar neighbor_value = -1;
        auto neighbor_idx_linear = sub2ind(linear_strides, neighbor_idx);
        bc(cell_idx, neighbor_idx, self_value, neighbor_value);
        entries.push_back({self_idx_linear, neighbor_idx_linear, neighbor_value});
      }

      neighbor_idx[i] -= 2;
      if (is_in_bound(dims_, neighbor_idx)) {
        Scalar neighbor_value = -1;
        auto neighbor_idx_linear = sub2ind(linear_strides, neighbor_idx);
        bc(cell_idx, neighbor_idx, self_value, neighbor_value);
        entries.push_back({self_idx_linear, neighbor_idx_linear, neighbor_value});
      }
    }

    entries.push_back({self_idx_linear, self_idx_linear, self_value});
  }

  const index_t total = dims_.numel();
  auto coo = make_from_triplets<Scalar>(entries.begin(), entries.end(), total, total, sparse_property::symmetric);

  if constexpr (SparseCompression == sparse_format::coo) {
    return coo;
  } else {
    return make_from_coos<Scalar, SparseCompression>(coo);
  }
}

}  // namespace mathprim::sparse