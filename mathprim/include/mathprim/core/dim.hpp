/**
 * @file
 * @brief Dim/Shape definition. We support up to N=4
 */

#pragma once

#include <type_traits>

#include "defines.hpp"
#include "utils/index_pack.hpp"

namespace mathprim {

/// @brief Fully dynamic shape.
template <index_t Ndim>
using dshape = god::apply_seq_t<index_pack, god::duplicate_t<Ndim, keep_dim>>;
/// @brief Fully dynamic stride.
template <index_t Ndim>
using dstride = god::apply_seq_t<index_pack, god::duplicate_t<Ndim, keep_dim>>;
template <index_t Ndim>
using dim_t = dshape<Ndim>;

///////////////////////////////////////////////////////////////////////////////
/// Enhance from pack to shape and strides
///////////////////////////////////////////////////////////////////////////////
namespace internal {

template <typename T>
struct is_index_pack : std::false_type {};
template <index_t... Svalues>
struct is_index_pack<index_pack<Svalues...>> : std::true_type {};

template <typename Seq>
struct default_stride;

template <>
struct default_stride<index_seq<>> {
  using type = index_seq<>;
};

template <index_t Single>
struct default_stride<index_seq<Single>> {
  using type = index_seq<1>;  // Continuous.
};

template <index_t Front, index_t... Args>
struct default_stride<index_seq<Front, Args...>> {
  using last_strides = typename default_stride<index_seq<Args...>>::type;
  static constexpr index_t last_stride = god::car<last_strides>::value;
  static constexpr index_t last_dim = god::car<index_seq<Args...>>::value;
  using type = typename god::prepend<(last_dim == keep_dim ? keep_dim : last_dim * last_stride), last_strides>::type;
};

// Aliases:
template <typename T>
constexpr bool is_index_pack_v = is_index_pack<T>::value;
template <typename Pack>
using default_stride_t = god::to_pack<typename default_stride<typename Pack::seq>::type>;

template <typename Shape, typename Stride>
constexpr bool is_contiguous_compile_time_v = internal::is_compile_time_equal_v<default_stride_t<Shape>, Stride>;

template <index_t... Svalues, index_t Ndim, index_t... Seq>
MATHPRIM_PRIMFUNC index_t sub2ind(const index_pack<Svalues...> &stride, const index_array<Ndim> &subscript,
                                  const index_seq<Seq...> & /* Seq */) noexcept {
  return ((stride.template get<Seq>() * subscript.template get<Seq>()) + ...);
}

template <index_t... Idx, typename L, typename R>
MATHPRIM_PRIMFUNC bool is_in_bound(const L &shape, const R &index, index_seq<Idx...>) noexcept {
  return ((index.template get<Idx>() >= 0 && index.template get<Idx>() < shape.template get<Idx>()) && ...);
}

template <index_t Svalue>
struct holder {
  MATHPRIM_PRIMFUNC index_t operator*() const noexcept {
    return Svalue;
  }
};

template <>
struct holder<keep_dim> {
  MATHPRIM_PRIMFUNC holder(index_t value) : value_(value) {}  // NOLINT
  MATHPRIM_PRIMFUNC index_t operator*() const noexcept {
    return value_;
  }
  index_t value_;
};

template <typename Integer>
struct can_hold : std::is_integral<Integer> {};
template <index_t Svalue>
struct can_hold<holder<Svalue>> : std::true_type {};
template <typename T>
constexpr bool can_hold_v = can_hold<T>::value;
template <typename T, bool IsIntegral = std::is_integral_v<std::decay_t<T>>>
struct to_holder_impl;
template <typename T>
struct to_holder_impl<T, true> {
  template <typename Integer>
  static MATHPRIM_PRIMFUNC holder<keep_dim> impl(Integer value) noexcept {
    return holder<keep_dim>{static_cast<index_t>(value)};
  }
  using type = holder<keep_dim>;
};

template <index_t Svalue>
struct to_holder_impl<holder<Svalue>, false> {
  static MATHPRIM_PRIMFUNC holder<Svalue> impl(const holder<Svalue> &value) noexcept {
    return value;
  }
  using type = holder<Svalue>;
};

template <typename T>
MATHPRIM_PRIMFUNC typename to_holder_impl<T>::type to_holder(T value) noexcept {
  return to_holder_impl<T>::impl(value);
}

template <typename... Args>
struct holders_to_shape_impl;
template <index_t FrontValue>
struct holders_to_shape_impl<holder<FrontValue>> {
  using type = index_seq<FrontValue>;
};
template <typename Front, typename... Args>
struct holders_to_shape_impl<Front, Args...> {
  using front_type = typename holders_to_shape_impl<Front>::type;
  using last_type = typename holders_to_shape_impl<Args...>::type;
  using type = god::prepend_t<god::car_v<front_type>, last_type>;
};
template <typename... Args>
using holders_to_shape_t = god::to_pack<typename holders_to_shape_impl<Args...>::type>;

template <typename... Holders>
MATHPRIM_PRIMFUNC holders_to_shape_t<Holders...> make_shape_from_holders(Holders... holders) {
  return holders_to_shape_t<Holders...>{(*holders)...};
}

template <index_t Base, index_t Alpha>
struct pow_impl {
  static_assert(Alpha >= 0, "The exponent must be non-negative.");
  static_assert(Base >= 0, "The base must be non-negative.");
  static constexpr index_t value = Base * pow_impl<Base, Alpha - 1>::value;
};
template <index_t Base>
struct pow_impl<Base, 0> {
  static constexpr index_t value = 1;
};

template <char... Args>
struct parse_int_impl;
template <char Head>
struct parse_int_impl<Head> {
  static_assert('0' <= Head && Head <= '9', "The input must be a digit.");
  static constexpr index_t value = Head - '0';
};
template <char Head, char... Tail>
struct parse_int_impl<Head, Tail...> {
  static constexpr index_t value = parse_int_impl<Tail...>::value + pow_impl<10, sizeof...(Tail)>::value * (Head - '0');
};

template <char... Args>
constexpr index_t parse_int = parse_int_impl<Args...>::value;


// batched
template <index_t... Svalues, index_t... Seq>
MATHPRIM_PRIMFUNC shape_t<keep_dim, Svalues...> batched_shape_impl(index_t batch_size, const shape_t<Svalues...> &shape,
                                                                   const index_seq<Seq...> & /* Seq */) noexcept {
  return shape_t<keep_dim, Svalues...>{batch_size, shape.template get<Seq>()...};
}

}  // namespace internal

using internal::holder;

template <typename... Args>
dshape<sizeof...(Args)> make_dshape(Args &&...args) noexcept {
  return dshape<sizeof...(Args)>{std::forward<Args>(args)...};
}

namespace literal {

/**
 * @brief Create a holder from a string literal integer.
 *
 * @tparam Args
 * @return MATHPRIM_PRIMFUNC constexpr
 */
template <char... Args>
MATHPRIM_PRIMFUNC constexpr auto operator""_s() {
  return internal::holder<::mathprim::internal::parse_int<Args...>>{};
}

}  // namespace literal

/**
 * @brief Create a shape from the given arguments.
 *
 * @tparam Args can be integer or holder.
 * @param args
 * @return shape_t<...> as static as possible, depending on the input.
 */
template <typename... Args>
MATHPRIM_PRIMFUNC auto make_shape(Args... args) {
  return internal::make_shape_from_holders(internal::to_holder<Args>(std::forward<Args>(args))...);
}

/**
 * @brief Default stride for a given shape, in bytes.
 *
 * @tparam T
 * @tparam pack
 */
template <typename Pack>
using default_stride_t = internal::default_stride_t<Pack>;

/**
 * @brief Calculate the byte offset from the subscript.
 *
 * @tparam T Scalar type.
 * @tparam Svalues
 * @param shape
 * @return default_stride_t<index_pack<Svalues...>>
 */
template <typename T, index_t... Svalues>
MATHPRIM_PRIMFUNC default_stride_t<index_pack<Svalues...>> make_default_stride(index_pack<Svalues...> shape) {
  using Ret = default_stride_t<index_pack<Svalues...>>;
  Ret stride;
  if constexpr (Ret::fully_static) {
    return stride;
  } else {
    stride[ndim(shape) - 1] = 1;
    for (index_t i = ndim(shape) - 2; i >= 0; --i) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
  }
}

template <index_t... Svalues, index_t Ndim>
MATHPRIM_PRIMFUNC index_t sub2ind(const stride_t<Svalues...> &stride, const index_array<Ndim> &subscript) noexcept {
  static_assert(Ndim == sizeof...(Svalues), "The subscript and stride must have the same dimension.");
  return internal::sub2ind(stride, subscript, make_index_seq<Ndim>{});
}

template <index_t... Svalues>
MATHPRIM_PRIMFUNC index_array<index_pack<Svalues...>::ndim> ind2sub(const shape_t<Svalues...> &shape,
                                                                    index_t index) noexcept {
  index_array<index_pack<Svalues...>::ndim> sub;
  index_t remaining = index;
  for (index_t i = ndim(shape) - 1; i >= 0; --i) {
    sub[i] = remaining % shape[i];
    remaining /= shape[i];
  }
  return sub;
}

/**
 * @brief Check if the index is in bound.
 *
 * @tparam Svalues
 * @param shape
 * @param index
 * @return MATHPRIM_PRIMFUNC
 */
template <index_t... Svalues>
MATHPRIM_PRIMFUNC bool is_in_bound(const shape_t<Svalues...> &shape,
                                   const index_array<sizeof...(Svalues)> &index) noexcept {
  return internal::is_in_bound(shape, index, make_index_seq<sizeof...(Svalues)>{});
}

template <index_t Ndim>
MATHPRIM_PRIMFUNC bool is_in_bound(const index_array<Ndim> &shape, const index_array<Ndim> &index) noexcept {
  return internal::is_in_bound(shape, index, make_index_seq<Ndim>{});
}

#define MATHPRIM_MAKE_SHAPE_VEC(dim)                                    \
  template <index_t... Svalues, typename... Integers>                   \
  shape_t<Svalues..., dim> make_shape_vec##dim(Integers &&...values) {  \
    return shape_t<Svalues..., dim>{std::forward<Integers>(values)...}; \
  }
#define MATHPRIM_MAKE_SHAPE_MAT(rows, cols)                                          \
  template <index_t... Svalues, typename... Integers>                                \
  shape_t<Svalues..., rows, cols> make_shape_mat##rows##cols(Integers &&...values) { \
    return shape_t<Svalues..., rows, cols>{std::forward<Integers>(values)...};       \
  }

MATHPRIM_MAKE_SHAPE_VEC(2);
MATHPRIM_MAKE_SHAPE_VEC(3);
MATHPRIM_MAKE_SHAPE_VEC(4);

MATHPRIM_MAKE_SHAPE_MAT(2, 2);
MATHPRIM_MAKE_SHAPE_MAT(2, 3);
MATHPRIM_MAKE_SHAPE_MAT(2, 4);
MATHPRIM_MAKE_SHAPE_MAT(3, 2);
MATHPRIM_MAKE_SHAPE_MAT(3, 3);
MATHPRIM_MAKE_SHAPE_MAT(3, 4);
MATHPRIM_MAKE_SHAPE_MAT(4, 2);
MATHPRIM_MAKE_SHAPE_MAT(4, 3);
MATHPRIM_MAKE_SHAPE_MAT(4, 4);

#undef MATHPRIM_MAKE_SHAPE_MAT
#undef MATHPRIM_MAKE_SHAPE_VEC

template <index_t... Svalues>
MATHPRIM_PRIMFUNC shape_t<keep_dim, Svalues...> batched_shape(index_t batch_size, const index_pack<Svalues...> &shape) noexcept {
  return internal::batched_shape_impl(batch_size, shape, make_index_seq<sizeof...(Svalues)>{});
}

}  // namespace mathprim
