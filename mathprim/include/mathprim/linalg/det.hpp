#pragma once
#include "mathprim/core/view.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim::linalg {

template <typename Scalar, typename Device, index_t Nrows>
struct small_det {
  using matrix_type = contiguous_view<const Scalar, shape_t<Nrows, Nrows>, Device>;

  MATHPRIM_PRIMFUNC Scalar operator()(matrix_type m) const noexcept {
    return eigen_support::cmap(m).determinant();
  }
};

// Partials for 2x2 and 3x3
template <typename Scalar, typename Device>
struct small_det<Scalar, Device, 2> {
  using matrix_type = contiguous_view<const Scalar, shape_t<2, 2>, Device>;

  MATHPRIM_PRIMFUNC Scalar operator()(matrix_type m) const noexcept {
    return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
  }
};

template <typename Scalar, typename Device>
struct small_det<Scalar, Device, 3> {
  using matrix_type = contiguous_view<const Scalar, shape_t<3, 3>, Device>;

  MATHPRIM_PRIMFUNC Scalar operator()(matrix_type m) const noexcept {
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -  //
           m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +  //
           m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
  }
};

}