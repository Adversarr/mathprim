#pragma once
#include "mathprim/core/view.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim::linalg {

template <typename Scalar, typename Device, index_t Nrows>
struct small_inv;

/**
 * @brief Computes the inverse of a small square matrix.
 * 
 * @tparam Scalar 
 * @tparam Device 
 */
template <typename Scalar, typename Device>
struct small_inv<Scalar, Device, 2> {
  using matrix_type = contiguous_view<Scalar, shape_t<2, 2>, Device>;
  using const_matrix = contiguous_view<const Scalar, shape_t<2, 2>, Device>;

  MATHPRIM_PRIMFUNC bool operator()(matrix_type dst, const_matrix src) const noexcept {
    const Scalar det_src = src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0);
    dst(0, 0) = +src(1, 1) / det_src;
    dst(0, 1) = -src(0, 1) / det_src;
    dst(1, 0) = -src(1, 0) / det_src;
    dst(1, 1) = +src(0, 0) / det_src;
    return det_src != 0;
  }
};

template <typename Scalar, typename Device>
struct small_inv<Scalar, Device, 3> {
  using matrix_type = contiguous_view<Scalar, shape_t<3, 3>, Device>;
  using const_matrix = contiguous_view<const Scalar, shape_t<3, 3>, Device>;

  MATHPRIM_PRIMFUNC bool operator()(matrix_type dst, const_matrix src) const noexcept {
    const Scalar det_src = src(0, 0) * (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1)) -
                           src(0, 1) * (src(1, 0) * src(2, 2) - src(1, 2) * src(2, 0)) +
                           src(0, 2) * (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0));
    dst(0, 0) = (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1)) / det_src;
    dst(0, 1) = (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2)) / det_src;
    dst(0, 2) = (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1)) / det_src;
    dst(1, 0) = (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2)) / det_src;
    dst(1, 1) = (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0)) / det_src;
    dst(1, 2) = (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2)) / det_src;
    dst(2, 0) = (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0)) / det_src;
    dst(2, 1) = (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1)) / det_src;
    dst(2, 2) = (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0)) / det_src;
    return det_src != 0;
  }
};

}  // namespace mathprim::linalg
