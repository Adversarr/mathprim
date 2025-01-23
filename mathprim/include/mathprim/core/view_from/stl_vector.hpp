#pragma once
#include <vector>

#include "mathprim/core/view.hpp"
#include "mathprim/core/utils/common.hpp"

namespace mathprim {

template <typename T>
basic_view<T, 1, device_t::cpu> view_from(
  std::vector<T>& vec) {
  return basic_view<T, 1, device_t::cpu>(vec.data(), vec.size());
}

template <typename T>
basic_view<const T, 1, device_t::cpu> view_from(
  const std::vector<T>& vec) {
  return basic_view<const T, 1, device_t::cpu>(vec.data(), vec.size());
}

template <typename T>
void view_from(std::vector<T>&& ) {
  static_assert(internal::always_false_v<T>, "view_from: only valid for l-value types");
}

}
