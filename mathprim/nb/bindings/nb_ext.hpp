#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <mathprim/core/view.hpp>

#include "mathprim/core/defines.hpp"

namespace nbex {
namespace nb = ::nanobind;
namespace mp = ::mathprim;

namespace internal {

using nb_index_t = nb::ssize_t;
using index_t = mp::index_t;

// convert shape/stride values.
template <index_t shape_value>
constexpr nb_index_t to_nb_index_v = shape_value == mp::keep_dim ? -1 : shape_value;
template <nb_index_t shape_value>
constexpr index_t to_mp_index_v = shape_value == -1 ? mp::keep_dim : shape_value;

// convert shape.
template <typename seq>
struct to_nb_shape;
template <index_t... svalues>
struct to_nb_shape<mp::index_pack<svalues...>> {
  using type = nb::shape<to_nb_index_v<svalues>...>;
};
template <typename seq>
using to_nb_shape_t = typename to_nb_shape<seq>::type;

template <typename nb_shape>
struct to_mp_shape;
template <nb_index_t... svalues>
struct to_mp_shape<nb::shape<svalues...>> {
  using type = mp::index_pack<to_mp_index_v<svalues>...>;
};
template <typename nb_shape>
using to_mp_shape_t = typename to_mp_shape<nb_shape>::type;

template <typename seq, index_t... idx>
std::array<size_t, seq::ndim> make_nb_shape_impl(const seq& shape, mp::index_seq<idx...>) {
  return {static_cast<size_t>(shape.template get<idx>())...};
}
template <typename seq>
std::array<size_t, seq::ndim> make_nb_shape(const seq& shape) {
  return make_nb_shape_impl(shape, mp::make_index_seq<seq::ndim>{});
}

// convert device.
template <typename mp_dev>
struct to_nb_device;
template <typename nb_dev>
struct to_mp_device;
#define MATHPRIM_INTERNAL_TO_NB_DEVICE(from, to, interface) \
  template <>                                               \
  struct to_nb_device<mp::device::from> {                   \
    using type = nb::device::to;                            \
    using api = nb::interface;                              \
  };                                                        \
  template <>                                               \
  struct to_mp_device<nb::device::to> {                     \
    using type = mp::device::from;                          \
    using api = nb::interface;                              \
  }

MATHPRIM_INTERNAL_TO_NB_DEVICE(cpu, cpu, numpy);
MATHPRIM_INTERNAL_TO_NB_DEVICE(cuda, cuda, pytorch);
#undef MATHEX_INTERNAL_TO_NB_DEVICE
template <typename mp_dev>
using to_nb_device_t = typename to_nb_device<mp_dev>::type;
template <typename nb_dev>
using to_mp_device_t = typename to_mp_device<nb_dev>::type;
template <typename mp_dev>
using to_nb_api_t = typename to_nb_device<mp_dev>::api;

// convert view
template <typename mp_view>
struct to_nb_array_standard;
template <typename T, index_t... sshape_values, index_t... sstride_values, typename dev>
struct to_nb_array_standard<mp::basic_view<T, mp::shape_t<sshape_values...>, mp::stride_t<sstride_values...>, dev>> {
  using view_t = mp::basic_view<T, mp::shape_t<sshape_values...>, mp::stride_t<sstride_values...>, dev>;
  using sshape = typename view_t::sshape;
  using nb_dev = to_nb_device_t<dev>;
  using nb_shape = to_nb_shape_t<sshape>;
  using nb_api = to_nb_api_t<dev>;
  using type = nb::ndarray<T, nb_shape, nb_dev, nb_api>;
};
template <typename mp_view>
using to_nb_array_standard_t = typename to_nb_array_standard<mp_view>::type;

template <typename T, typename sshape, typename sstride, typename dev>
to_nb_array_standard_t<mp::basic_view<T, sshape, sstride, dev>> make_nb_array_standard(
    mp::basic_view<T, sshape, sstride, dev> view) {
  using ret_t = to_nb_array_standard_t<mp::basic_view<T, sshape, sstride, dev>>;
  auto shape = make_nb_shape(view.shape());
  return ret_t(view.data(), shape.size(), shape.data());
}

template <typename nb_view>
struct to_mp_view_standard;
template <typename... Args>
struct to_mp_view_standard<nb::ndarray<Args...>> {
  using nb_view = nb::ndarray<Args...>;
  using Config = typename nb_view::Config;
  using Shape = typename Config::Shape;
  using Scalar = typename nb_view::Scalar;

  using mp_shape = to_mp_shape_t<Shape>;
  using mp_dev = to_mp_device_t<typename Config::DeviceType>;
  using type = mp::continuous_view<Scalar, mp_shape, mp_dev>;
};
template <typename nb_view>
using to_mp_view_standard_t = typename to_mp_view_standard<nb_view>::type;

template <typename... Args>
to_mp_view_standard_t<nb::ndarray<Args...>> make_mp_view_standard(nb::ndarray<Args...> view) {
  using ret_t = to_mp_view_standard_t<nb::ndarray<Args...>>;
  using sshape = typename ret_t::sshape;
  sshape shape;

  for (index_t i = 0; i < sshape::ndim; ++i) {
    shape[i] = view.shape(i);
  }
  return ret_t(view.data(), shape);
}

}  // namespace internal

template <typename T, typename sshape, typename sstride, typename dev>
auto to_nb_array_standard(mp::basic_view<T, sshape, sstride, dev> view) {
  return internal::make_nb_array_standard(view);
}

template <typename nb_view>
auto to_mp_view_standard(nb_view view) {
  return internal::make_mp_view_standard(view);
}

}  // namespace nbex
