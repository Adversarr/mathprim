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

}  // namespace mathprim::functional
