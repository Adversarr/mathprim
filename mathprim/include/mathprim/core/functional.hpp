#pragma once
#include "mathprim/core/view.hpp"  // IWYU pragma: export

namespace mathprim::functional {

template <typename Dst, typename Lhs, typename Rhs, bool IsScalar = std::is_arithmetic_v<Dst>>
struct plus;

template <typename Dst, typename Lhs, typename Rhs>
struct plus<Dst, Lhs, Rhs, false> {
  MATHPRIM_PRIMFUNC void operator()(Dst dst, Lhs lhs, Rhs rhs) const noexcept {
    MATHPRIM_ASSERT(lhs.shape() == rhs.shape() && lhs.shape() == dst.shape());
    for (auto i : dst.shape()) {
      dst(i) = lhs(i) + rhs(i);
    }
  }
};

template <typename Dst, typename Lhs, typename Rhs>
struct plus<Dst, Lhs, Rhs, true> {
  MATHPRIM_PRIMFUNC void operator()(Dst& dst, const Lhs& lhs, const Rhs& rhs) const noexcept { dst = lhs + rhs; }
};

template <typename Dst, typename Src, bool IsScalar = std::is_arithmetic_v<Dst>>
struct madd;

template <typename Dst, typename Src>
struct madd<Dst, Src, false> {
  using scalar_type = typename Dst::scalar_type;
  scalar_type alpha_ = 1;
  madd() = default;
  MATHPRIM_PRIMFUNC explicit madd(scalar_type alpha) noexcept : alpha_(alpha) {}
  madd(const madd&) = default;
  madd& operator=(const madd&) = default;

  MATHPRIM_PRIMFUNC void operator()(Dst dst, Src src) const noexcept {
    MATHPRIM_ASSERT(dst.shape() == src.shape());
    for (auto i : dst.shape()) {
      dst(i) += alpha_ * src(i);
    }
  }
};

template <typename Dst, typename Src>
struct madd<Dst, Src, true> {
  static_assert(std::is_same_v<Dst, Src>, "Expect same arithmetic type for Dst and Src");
  Dst alpha_ = 1;
  madd() = default;
  MATHPRIM_PRIMFUNC explicit madd(Dst alpha) noexcept : alpha_(alpha) {}
  madd(const madd&) = default;
  madd& operator=(const madd&) = default;

  MATHPRIM_PRIMFUNC void operator()(Dst& dst, const Src& src) const noexcept { dst += alpha_ * src; }
};

template <typename Scalar, typename Device, index_t Ndim>
struct set_vector {
  Scalar values_[Ndim]{0};
  MATHPRIM_PRIMFUNC set_vector() = default;
  MATHPRIM_INTERNAL_COPY(set_vector, default);

  template <typename Sstride>
  MATHPRIM_PRIMFUNC explicit set_vector(const basic_view<const Scalar, shape_t<Ndim>, Sstride, Device>& values) {
    for (index_t i = 0; i < Ndim; ++i) {
      values_[i] = values(i);
    }
  }

  template <typename Sstride>
  MATHPRIM_PRIMFUNC void operator()(const basic_view<Scalar, shape_t<Ndim>, Sstride, Device>& dst) const {
    for (index_t i = 0; i < Ndim; ++i) {
      dst[i] = values_[i];
    }
  }
};

template <typename Scalar, typename Device, index_t Ndim>
struct affine_transform {
public:
  Scalar lin_[Ndim * Ndim]{0};
  Scalar bias_[Ndim]{0};
  MATHPRIM_PRIMFUNC affine_transform() = default;
  MATHPRIM_INTERNAL_COPY(affine_transform, default);

  template <typename Sstride, typename Device2>
  MATHPRIM_PRIMFUNC explicit affine_transform(
      const basic_view<const Scalar, shape_t<Ndim, Ndim>, Sstride, Device2>& lin) {
    for (auto [i, j] : lin.shape()) {
      lin_[i * Ndim + j] = lin(i, j);
    }

    for (index_t i = 0; i < Ndim; ++i) {
      bias_[i] = 0;
    }
  }
  template <typename SstrideLin, typename DeviceLin, typename SstrideBias, typename DeviceBias>
  MATHPRIM_PRIMFUNC affine_transform(const basic_view<const Scalar, shape_t<Ndim, Ndim>, SstrideLin, DeviceLin>& lin,
                                     const basic_view<const Scalar, shape_t<Ndim>, SstrideBias, DeviceBias>& bias) {
    for (auto [i, j] : lin.shape()) {
      lin_[i * Ndim + j] = lin(i, j);
    }

    for (index_t i = 0; i < Ndim; ++i) {
      bias_[i] = bias(i);
    }
  }

  template <typename SstrideSrc, typename SstrideDst>
  MATHPRIM_PRIMFUNC void operator()(const basic_view<Scalar, shape_t<Ndim>, SstrideDst, Device>& dst,
                                    const basic_view<const Scalar, shape_t<Ndim>, SstrideSrc, Device>& src) const {
    for (index_t i = 0; i < Ndim; ++i) {
      dst[i] = bias_[i];
      for (index_t j = 0; j < Ndim; ++j) {
        dst[i] += lin_[i * Ndim + j] * src[j];
      }
    }
  }

  template <typename SstrideDst>
  MATHPRIM_PRIMFUNC void operator()(const basic_view<Scalar, shape_t<Ndim>, SstrideDst, Device>& dst) const {
    Scalar out[Ndim]{0};
    for (index_t i = 0; i < Ndim; ++i) {
      out[i] = bias_[i];
      for (index_t j = 0; j < Ndim; ++j) {
        out[i] += lin_[i * Ndim + j] * dst[j];
      }
    }
    for (index_t i = 0; i < Ndim; ++i) {
      dst[i] = out[i];
    }
  }
};

}  // namespace mathprim::functional
