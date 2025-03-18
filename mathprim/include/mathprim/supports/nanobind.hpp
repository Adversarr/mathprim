#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <array>

#include "mathprim/core/view.hpp"

#define TOSTR(x) #x
namespace nb = ::nanobind;
namespace mp = ::mathprim;

namespace nbex {

namespace internal {

using nb_index_t = ssize_t;
using index_t = mp::index_t;

// convert shape/stride values.
template <index_t ShapeValue>
constexpr nb_index_t to_nb_index_v = ShapeValue == mp::keep_dim ? -1 : ShapeValue;
template <nb_index_t ShapeValue>
constexpr index_t to_mp_index_v = ShapeValue == -1 ? mp::keep_dim : ShapeValue;

// convert shape.
template <typename Seq>
struct to_nb_shape;
template <index_t... Svalues>
struct to_nb_shape<mp::index_pack<Svalues...>> {
  using type = nb::shape<to_nb_index_v<Svalues>...>;
};
template <typename Seq>
using to_nb_shape_t = typename to_nb_shape<Seq>::type;

template <typename NbShape>
struct to_mp_shape;
template <nb_index_t... Svalues>
struct to_mp_shape<nb::shape<Svalues...>> {
  using type = mp::index_pack<to_mp_index_v<Svalues>...>;
};
template <typename NbShape>
using to_mp_shape_t = typename to_mp_shape<NbShape>::type;

template <typename Seq, index_t... Idx>
std::array<size_t, Seq::ndim> make_nb_shape_impl(const Seq& shape, mp::index_seq<Idx...>) {
  return {static_cast<size_t>(shape.template get<Idx>())...};
}
template <typename Seq>
std::array<size_t, Seq::ndim> make_nb_shape(const Seq& shape) {
  return make_nb_shape_impl(shape, mathprim::internal::make_index_seq<Seq::ndim>{});
}

// convert device.
template <typename MpDev>
struct to_nb_device;
template <typename NbDev>
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
template <typename MpDev>
using to_nb_device_t = typename to_nb_device<MpDev>::type;
template <typename NbDev>
using to_mp_device_t = typename to_mp_device<NbDev>::type;
template <typename MpDev>
using to_nb_api_t = typename to_nb_device<MpDev>::api;

// convert view
template <typename MpView>
struct to_nb_array_standard;
template <typename T, index_t... SshapeValues, index_t... SstrideValues, typename Dev>
struct to_nb_array_standard<mp::basic_view<T, mp::shape_t<SshapeValues...>, mp::stride_t<SstrideValues...>, Dev>> {
  using view_t = mp::basic_view<T, mp::shape_t<SshapeValues...>, mp::stride_t<SstrideValues...>, Dev>;
  using sshape = typename view_t::shape_at_compile_time;
  using nb_dev = to_nb_device_t<Dev>;
  using nb_shape = to_nb_shape_t<sshape>;
  using nb_api = to_nb_api_t<Dev>;
  using type = nb::ndarray<T, nb_shape, nb_dev, nb_api>;
};

template <typename MpView>
using to_nb_array_standard_t = typename to_nb_array_standard<MpView>::type;

template <typename T, typename Sshape, typename Sstride, typename Dev>
to_nb_array_standard_t<mp::basic_view<T, Sshape, Sstride, Dev>> make_nb_array_standard(
    mp::basic_view<T, Sshape, Sstride, Dev> view) {
  using ret_t = to_nb_array_standard_t<mp::basic_view<T, Sshape, Sstride, Dev>>;
  auto shape = make_nb_shape(view.shape());
  return ret_t(view.data(), shape.size(), shape.data());
}

template <typename NbView>
struct to_mp_view_standard;
template <typename... Args>
struct to_mp_view_standard<nb::ndarray<Args...>> {
  using nb_view = nb::ndarray<Args...>;
  using Config = typename nb_view::Config;
  using Shape = typename Config::Shape;
  using Scalar = typename nb_view::Scalar;

  using mp_shape = to_mp_shape_t<Shape>;
  using mp_dev = to_mp_device_t<typename Config::DeviceType>;
  using type = mp::contiguous_view<Scalar, mp_shape, mp_dev>;
};
template <typename NbView>
using to_mp_view_standard_t = typename to_mp_view_standard<NbView>::type;

template <typename... Args>
to_mp_view_standard_t<nb::ndarray<Args...>> make_mp_view_standard(nb::ndarray<Args...> view) {
  using ret_t = to_mp_view_standard_t<nb::ndarray<Args...>>;
  using sshape = typename ret_t::shape_at_compile_time;
  sshape shape;

  for (index_t i = 0; i < sshape::ndim; ++i) {
    shape[i] = view.shape(i);
  }
  return ret_t(view.data(), shape);
}

}  // namespace internal

template <typename T, typename Sshape, typename Sstride, typename Dev>
auto to_nb_array_standard(mp::basic_view<T, Sshape, Sstride, Dev> view) {
  return internal::make_nb_array_standard(view);
}

template <typename NbView>
auto to_mp_view_standard(NbView view) {
  return internal::make_mp_view_standard(view);
}

}  // namespace nbex
