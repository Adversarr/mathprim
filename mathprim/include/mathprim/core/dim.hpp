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
template <index_t ndim>
using dshape = god::apply_seq_t<index_pack, god::duplicate_t<ndim, keep_dim>>;
/// @brief Fully dynamic stride.
template <index_t ndim>
using dstride = god::apply_seq_t<index_pack, god::duplicate_t<ndim, keep_dim>>;
template <index_t ndim>
using dim_t = dshape<ndim>;

///////////////////////////////////////////////////////////////////////////////
/// Enhance from pack to shape and strides
///////////////////////////////////////////////////////////////////////////////
namespace internal {

template <typename T>
struct is_index_pack : std::false_type {};
template <index_t... svalues>
struct is_index_pack<index_pack<svalues...>> : std::true_type {};

template <typename seq>
struct default_stride;

template <>
struct default_stride<index_seq<>> {
  using type = index_seq<>;
};

template <index_t single>
struct default_stride<index_seq<single>> {
  using type = index_seq<1>;  // Continuous.
};

template <index_t front, index_t... args>
struct default_stride<index_seq<front, args...>> {
  using last_strides = typename default_stride<index_seq<args...>>::type;
  static constexpr index_t last_stride = god::car<last_strides>::value;
  static constexpr index_t last_dim = god::car<index_seq<args...>>::value;
  using type = typename god::prepend<(last_dim == keep_dim ? keep_dim : last_dim * last_stride), last_strides>::type;
};

// Aliases:
template <typename T>
constexpr bool is_index_pack_v = is_index_pack<T>::value;
template <typename pack>
using default_stride_t = god::to_pack<typename default_stride<typename pack::seq>::type>;

template <typename shape, typename stride>
constexpr bool is_contiguous_compile_time_v = internal::is_compile_time_equal_v<default_stride_t<shape>, stride>;

template <index_t... svalues, index_t ndim, index_t... seq>
MATHPRIM_PRIMFUNC index_t sub2ind(const index_pack<svalues...> &stride, const index_array<ndim> &subscript,
                                  const index_seq<seq...> & /* seq */) noexcept {
  return ((stride.template get<seq>() * subscript.template get<seq>()) + ...);
}

template <index_t... idx, typename L, typename R>
MATHPRIM_PRIMFUNC bool is_in_bound(const L &shape, const R &index, index_seq<idx...>) noexcept {
  return ((index.template get<idx>() >= 0 && index.template get<idx>() < shape.template get<idx>()) && ...);
}

template <index_t svalue>
struct holder {
  MATHPRIM_PRIMFUNC index_t operator*() const noexcept {
    return svalue;
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
template <index_t svalue>
struct can_hold<holder<svalue>> : std::true_type {};
template <typename T>
constexpr bool can_hold_v = can_hold<T>::value;
template <typename T, bool is_integral = std::is_integral_v<std::decay_t<T>>>
struct to_holder_impl;
template <typename T>
struct to_holder_impl<T, true> {
  template <typename Integer>
  static MATHPRIM_PRIMFUNC holder<keep_dim> impl(Integer value) noexcept {
    return holder<keep_dim>{static_cast<index_t>(value)};
  }
  using type = holder<keep_dim>;
};

template <index_t svalue>
struct to_holder_impl<holder<svalue>, false> {
  static MATHPRIM_PRIMFUNC holder<svalue> impl(const holder<svalue> &value) noexcept {
    return value;
  }
  using type = holder<svalue>;
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

template <index_t base, index_t alpha>
struct pow_impl {
  static_assert(alpha >= 0, "The exponent must be non-negative.");
  static_assert(base >= 0, "The base must be non-negative.");
  static constexpr index_t value = base * pow_impl<base, alpha - 1>::value;
};
template <index_t base>
struct pow_impl<base, 0> {
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
template <index_t... svalues, index_t... seq>
MATHPRIM_PRIMFUNC shape_t<keep_dim, svalues...> batched_shape_impl(index_t batch_size, const shape_t<svalues...> &shape,
                                                                   const index_seq<seq...> & /* seq */) noexcept {
  return shape_t<keep_dim, svalues...>{batch_size, shape.template get<seq>()...};
}

}  // namespace internal

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
template <typename pack>
using default_stride_t = internal::default_stride_t<pack>;

/**
 * @brief Calculate the byte offset from the subscript.
 *
 * @tparam T Scalar type.
 * @tparam svalues
 * @param shape
 * @return default_stride_t<index_pack<svalues...>>
 */
template <typename T, index_t... svalues>
MATHPRIM_PRIMFUNC default_stride_t<index_pack<svalues...>> make_default_stride(index_pack<svalues...> shape) {
  using Ret = default_stride_t<index_pack<svalues...>>;
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

template <index_t... svalues, index_t ndim>
MATHPRIM_PRIMFUNC index_t sub2ind(const stride_t<svalues...> &stride, const index_array<ndim> &subscript) noexcept {
  static_assert(ndim == sizeof...(svalues), "The subscript and stride must have the same dimension.");
  return internal::sub2ind(stride, subscript, make_index_seq<ndim>{});
}

template <index_t... svalues>
MATHPRIM_PRIMFUNC index_array<index_pack<svalues...>::ndim> ind2sub(const shape_t<svalues...> &shape,
                                                                    index_t index) noexcept {
  index_array<index_pack<svalues...>::ndim> sub;
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
 * @tparam svalues
 * @param shape
 * @param index
 * @return MATHPRIM_PRIMFUNC
 */
template <index_t... svalues>
MATHPRIM_PRIMFUNC bool is_in_bound(const shape_t<svalues...> &shape,
                                   const index_array<sizeof...(svalues)> &index) noexcept {
  return internal::is_in_bound(shape, index, make_index_seq<sizeof...(svalues)>{});
}

template <index_t ndim>
MATHPRIM_PRIMFUNC bool is_in_bound(const index_array<ndim> &shape, const index_array<ndim> &index) noexcept {
  return internal::is_in_bound(shape, index, make_index_seq<ndim>{});
}

#define MATHPRIM_MAKE_SHAPE_VEC(dim)                                    \
  template <index_t... svalues, typename... Integers>                   \
  shape_t<svalues..., dim> make_shape_vec##dim(Integers &&...values) {  \
    return shape_t<svalues..., dim>{std::forward<Integers>(values)...}; \
  }
#define MATHPRIM_MAKE_SHAPE_MAT(rows, cols)                                          \
  template <index_t... svalues, typename... Integers>                                \
  shape_t<svalues..., rows, cols> make_shape_mat##rows##cols(Integers &&...values) { \
    return shape_t<svalues..., rows, cols>{std::forward<Integers>(values)...};       \
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
