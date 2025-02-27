#pragma once
#include "mathprim/core/view.hpp"
namespace mathprim::geometry {
template <typename Scalar, index_t SpaceNdim, index_t SimplexNdim, typename Device>
struct basic_mesh {
  using vertices = contiguous_view<Scalar, shape_t<keep_dim, SpaceNdim>, Device>;
  using indices = contiguous_view<index_t, shape_t<keep_dim, SimplexNdim>, Device>;
  vertices vertices_;
  indices indices_;
};
}  // namespace mathprim::geometry