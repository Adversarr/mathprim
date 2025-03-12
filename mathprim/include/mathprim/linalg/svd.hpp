#pragma once
#include "mathprim/core/view.hpp"
#include "mathprim/linalg/det.hpp"
#include "mathprim/supports/eigen_dense.hpp"
// This file should be included last?
#include "./internal/svd3.h"

namespace mathprim::linalg {

////////////////////////////////////////////////
/// Singular Value Decompisition
////////////////////////////////////////////////
template <typename Scalar, typename Device>
struct svd_apply_rotation_variant;

template <typename Scalar, typename Device, index_t Nrows, index_t Ncols, bool Exact = true>
struct small_svd {
  static constexpr index_t singular_size = std::min(Nrows, Ncols);
  using matrix_type = contiguous_view<const Scalar, shape_t<Nrows, Ncols>, Device>;
  using matrix_u_type = contiguous_view<Scalar, shape_t<Nrows, Nrows>, Device>;
  using matrix_v_type = contiguous_view<Scalar, shape_t<Ncols, Ncols>, Device>;
  using singular_type = contiguous_view<Scalar, shape_t<singular_size>, Device>;

  MATHPRIM_PRIMFUNC void operator()(  //
      const matrix_type& m,           //
      const matrix_u_type& u,         //
      const matrix_v_type& v,         //
      const singular_type& sigma) const noexcept {
    Eigen::Matrix<Scalar, Nrows, Ncols> mat_a = eigen_support::cmap(m).transpose();
    auto mat_u = eigen_support::cmap(u);
    auto mat_v = eigen_support::cmap(v);
    auto vec_s = eigen_support::cmap(sigma);
    auto svd = mat_a.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    mat_u.transpose() = svd.matrixU();
    mat_v.transpose() = svd.matrixV();
    vec_s = svd.singularValues();
  }
};

template <typename Device>
struct small_svd<float, Device, 3, 3, true> {
  using matrix_type = contiguous_view<const float, shape_t<3, 3>, Device>;
  using matrix_u_type = contiguous_view<float, shape_t<3, 3>, Device>;
  using matrix_v_type = contiguous_view<float, shape_t<3, 3>, Device>;
  using singular_type = contiguous_view<float, shape_t<3>, Device>;

  bool rotation_variant_ = false;
  small_svd() = default;
  MATHPRIM_PRIMFUNC explicit small_svd(bool rotation_variant = false) : rotation_variant_(rotation_variant) {}
  small_svd(const small_svd&) = default;
  small_svd& operator=(const small_svd&) = default;
  small_svd(small_svd&&) = default;
  small_svd& operator=(small_svd&&) = default;

  MATHPRIM_PRIMFUNC void operator()(  //
      const matrix_type& m,           //
      const matrix_u_type& u,         //
      const matrix_v_type& v,         //
      const singular_type& sigma) const noexcept {
    ::mathprim::internal::svd(                                                            //
        m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2), m(2, 0), m(2, 1), m(2, 2),  //
        u(0, 0), u(0, 1), u(0, 2), u(1, 0), u(1, 1), u(1, 2), u(2, 0), u(2, 1), u(2, 2),  //
        sigma(0), sigma(1), sigma(2),                                                     //
        v(0, 0), v(0, 1), v(0, 2), v(1, 0), v(1, 1), v(1, 2), v(2, 0), v(2, 1), v(2, 2));
    if (rotation_variant_) {
      svd_apply_rotation_variant<float, Device>{}(u, v, sigma);
    }
  }
};

template <typename Device>
struct small_svd<float, Device, 3, 3, false> {
  using matrix_type = contiguous_view<const float, shape_t<3, 3>, Device>;
  using matrix_u_type = contiguous_view<float, shape_t<3, 3>, Device>;
  using matrix_v_type = contiguous_view<float, shape_t<3, 3>, Device>;
  using singular_type = contiguous_view<float, shape_t<3>, Device>;

  bool rotation_variant_ = false;
  small_svd() = default;
  MATHPRIM_PRIMFUNC explicit small_svd(bool rotation_variant = false) : rotation_variant_(rotation_variant) {}
  small_svd(const small_svd&) = default;
  small_svd& operator=(const small_svd&) = default;
  small_svd(small_svd&&) = default;
  small_svd& operator=(small_svd&&) = default;

  MATHPRIM_PRIMFUNC void operator()(  //
      const matrix_type& m,           //
      const matrix_u_type& u,         //
      const matrix_v_type& v,         //
      const singular_type& sigma) const noexcept {
    ::mathprim::internal::svd(                                                            //
        m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2), m(2, 0), m(2, 1), m(2, 2),  //
        u(0, 0), u(0, 1), u(0, 2), u(1, 0), u(1, 1), u(1, 2), u(2, 0), u(2, 1), u(2, 2),  //
        sigma(0), sigma(1), sigma(2),                                                     //
        v(0, 0), v(0, 1), v(0, 2), v(1, 0), v(1, 1), v(1, 2), v(2, 0), v(2, 1), v(2, 2));
    if (rotation_variant_) {
      svd_apply_rotation_variant<float, Device>{}(u, v, sigma);
    }
  }
};


template <typename Device>
struct small_svd<double, Device, 3, 3, false> {
  using matrix_type = contiguous_view<const double, shape_t<3, 3>, Device>;
  using matrix_u_type = contiguous_view<double, shape_t<3, 3>, Device>;
  using matrix_v_type = contiguous_view<double, shape_t<3, 3>, Device>;
  using singular_type = contiguous_view<double, shape_t<3>, Device>;

  bool rotation_variant_ = false;
  small_svd() = default;
  MATHPRIM_PRIMFUNC explicit small_svd(bool rotation_variant = false) : rotation_variant_(rotation_variant) {}
  small_svd(const small_svd&) = default;
  small_svd& operator=(const small_svd&) = default;
  small_svd(small_svd&&) = default;
  small_svd& operator=(small_svd&&) = default;

  MATHPRIM_PRIMFUNC void operator()(  //
      const matrix_type& m,           //
      const matrix_u_type& u,         //
      const matrix_v_type& v,         //
      const singular_type& sigma) const noexcept {
    float a[9] = {static_cast<float>(m(0, 0)), static_cast<float>(m(0, 1)), static_cast<float>(m(0, 2)),
                  static_cast<float>(m(1, 0)), static_cast<float>(m(1, 1)), static_cast<float>(m(1, 2)),
                  static_cast<float>(m(2, 0)), static_cast<float>(m(2, 1)), static_cast<float>(m(2, 2))};
    float u_32[9], v_32[9], sigma_32[3];
    ::mathprim::internal::svd(                                                            //
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],                             //
        u_32[0], u_32[1], u_32[2], u_32[3], u_32[4], u_32[5], u_32[6], u_32[7], u_32[8],  //
        sigma_32[0], sigma_32[1], sigma_32[2],                                            //
        v_32[0], v_32[1], v_32[2], v_32[3], v_32[4], v_32[5], v_32[6], v_32[7], v_32[8]);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        u(i, j) = static_cast<double>(u_32[i * 3 + j]);
        v(i, j) = static_cast<double>(v_32[i * 3 + j]);
      }
      sigma(i) = static_cast<double>(sigma_32[i]);
    }
    if (rotation_variant_) {
      svd_apply_rotation_variant<float, Device>{}(u, v, sigma);
    }
  }
};

/**
 * @brief For a 3x3 matrix, apply rotation variant.
 * 
 * @tparam Scalar 
 * @tparam Device 
 */
template <typename Scalar, typename Device>
struct svd_apply_rotation_variant {
  using matrix_type = contiguous_view<const Scalar, shape_t<3, 3>, Device>;
  using matrix_u_type = contiguous_view<Scalar, shape_t<3, 3>, Device>;
  using matrix_v_type = contiguous_view<Scalar, shape_t<3, 3>, Device>;
  using singular_type = contiguous_view<Scalar, shape_t<3>, Device>;

  // see Dynamic Deformables. Page 227.
  MATHPRIM_PRIMFUNC bool operator()(const matrix_u_type& u,  //
                                    const matrix_v_type& v,  //
                                    const singular_type& sigma) const noexcept {
    bool has_rotation = false;
    small_det<Scalar, Device, 3> det;
    auto det_u = det(u);
    auto det_v = det(v);
    auto l = det_u * det_v;
    if (det_u < 0 && det_v > 0) {
      u(0, 1) *= l;
      u(1, 1) *= l;
      u(2, 1) *= l;
      has_rotation = true;
    } else if (det_u > 0 && det_v < 0) {
      v(0, 1) *= l;
      v(1, 1) *= l;
      v(2, 1) *= l;
      has_rotation = true;
    }
    sigma(1) *= l;
    return has_rotation;
  }
};


}  // namespace mathprim::linalg
