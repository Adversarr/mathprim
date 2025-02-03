/**
 * @file
 * @brief Dim/Shape definition. We support up to N=4
 */

#pragma once

#include "defines.hpp"
#include "utils/index_pack.hpp"

namespace mathprim {

///////////////////////////////////////////////////////////////////////////////
/// Enhance from pack to shape and strides
///////////////////////////////////////////////////////////////////////////////
namespace internal {

template <typename T>
struct is_index_pack : std::false_type {};
template <index_t... svalues>
struct is_index_pack<index_pack<svalues...>> : std::true_type {};

template <index_t ndim>
using dim_t = index_array<ndim>;

template <typename T, typename seq>
struct default_stride;
template <typename T, index_t single>
struct default_stride<T, index_seq<single>> {
  static_assert((sizeof(T) > static_cast<size_t>(0)), "The type must have a size.");
  using type = index_seq<static_cast<index_t>(sizeof(T))>;  // Continuous.
};

template <typename T, index_t front, index_t... args>
struct default_stride<T, index_seq<front, args...>> {
  using last_strides = typename default_stride<T, index_seq<args...>>::type;
  static constexpr index_t last_stride = god::car<last_strides>::value;
  static constexpr index_t last_dim = god::car<index_seq<args...>>::value;
  using type = typename god::prepend<(last_dim == keep_dim ? keep_dim : last_dim * last_stride), last_strides>::type;
};

// Aliases:
template <typename T>
constexpr bool is_index_pack_v = is_index_pack<T>::value;
template <typename T, typename pack>
using default_stride_t = god::to_pack<typename default_stride<T, typename pack::seq>::type>;

template <typename T, typename shape, typename stride>
constexpr bool is_continuous_compile_time_v = internal::is_compile_time_equal_v<default_stride_t<T, shape>, stride>;

template <index_t... svalues, index_t ndim, index_t... seq>
MATHPRIM_PRIMFUNC index_t byte_offset(const index_pack<svalues...> &stride, const index_array<ndim> &subscript,
                                      const index_seq<seq...> & /* seq */) noexcept {
  return ((stride.template get<seq>() * subscript.template get<seq>()) + ...);
}

template <index_t... idx, typename L, typename R>
MATHPRIM_PRIMFUNC bool is_in_bound(const L &shape, const R &index, index_seq<idx...>) noexcept {
  return ((index.template get<idx>() >= 0 && index.template get<idx>() < shape.template get<idx>()) && ...);
}

}  // namespace internal

template <index_t ndim>
using dynamic_shape = god::apply_seq_t<index_pack, god::duplicate_t<ndim, keep_dim>>;
template <index_t ndim>
using dynamic_stride = god::apply_seq_t<index_pack, god::duplicate_t<ndim, keep_dim>>;

template <typename T, index_t... svalues>
MATHPRIM_PRIMFUNC internal::default_stride_t<T, index_pack<svalues...>> make_default_stride(
    index_pack<svalues...> shape) {
  using Ret = internal::default_stride_t<T, index_pack<svalues...>>;
  Ret stride;
  if constexpr (Ret::fully_static) {
    return stride;
  } else {
    stride[ndim(shape) - 1] = sizeof(T);
    for (index_t i = ndim(shape) - 2; i >= 0; --i) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
  }
}

template <index_t... svalues, index_t ndim>
MATHPRIM_PRIMFUNC index_t byte_offset(const stride_t<svalues...> &stride, const index_array<ndim> &subscript) noexcept {
  static_assert(ndim == sizeof...(svalues), "The subscript and stride must have the same dimension.");
  return internal::byte_offset(stride, subscript, make_index_seq<ndim>{});
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

template <index_t... svalues>
MATHPRIM_PRIMFUNC bool is_in_bound(const shape_t<svalues...> &shape,
                                   const index_array<sizeof...(svalues)> &index) noexcept {
  return internal::is_in_bound(shape, index, make_index_seq<sizeof...(svalues)>{});
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

template <typename... Args>
dynamic_shape<sizeof...(Args)> make_dynamic_shape(Args &&...args) noexcept {
  return dynamic_shape<sizeof...(Args)>{std::forward<Args>(args)...};
}

}  // namespace mathprim
