#pragma once
#include <array>
#include <vector>

#include "mathprim/core/view.hpp"
namespace mathprim {

template <typename T>
contiguous_view<const T, shape_t<-1>, device::cpu> view(const std::vector<T>& vec) {
  using sshape = shape_t<-1>;
  sshape shape(static_cast<index_t>(vec.size()));
  return contiguous_view<T, sshape, device::cpu>(vec.data(), shape);
}
template <typename T>
contiguous_view<T, shape_t<-1>, device::cpu> view(std::vector<T>& vec) {
  using sshape = shape_t<-1>;
  sshape shape(static_cast<index_t>(vec.size()));
  return contiguous_view<T, sshape, device::cpu>(vec.data(), shape);
}

template <typename T, index_t N>
contiguous_view<const T, shape_t<N>, device::cpu> view(const std::array<T, N>& arr) {
  using sshape = shape_t<N>;
  return contiguous_view<T, sshape, device::cpu>(arr.data(), sshape());
}

template <typename T, index_t N>
contiguous_view<T, shape_t<N>, device::cpu> view(std::array<T, N>& arr) {
  using sshape = shape_t<N>;
  return contiguous_view<T, sshape, device::cpu>(arr.data(), sshape());
}

template <typename T, index_t N>
contiguous_view<T, shape_t<N>, device::cpu> view(T (&arr)[N]) {
  using sshape = shape_t<N>;
  return contiguous_view<T, sshape, device::cpu>(arr, sshape());
}

template <typename T, index_t N>
contiguous_view<const T, shape_t<N>, device::cpu> view(const T (&arr)[N]) {
  using sshape = shape_t<N>;
  return contiguous_view<T, sshape, device::cpu>(arr, sshape());
}

}  // namespace mathprim
