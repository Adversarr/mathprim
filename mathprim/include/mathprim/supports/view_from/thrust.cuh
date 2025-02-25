#pragma once
#include <thrust/device_vector.h>

#include "mathprim/core/view.hpp"

namespace mathprim {

template <typename T>
contiguous_view<const T, shape_t<-1>, device::cpu> view(const thrust::device_vector<T>& vec) {
  using sshape = shape_t<-1>;
  sshape shape(static_cast<index_t>(vec.size()));
  auto ptr = thrust::raw_pointer_cast(vec.data());
  return contiguous_view<T, sshape, device::cpu>(ptr, shape);
}
template <typename T>
contiguous_view<T, shape_t<-1>, device::cpu> view(thrust::device_vector<T>& vec) {
  using sshape = shape_t<-1>;
  sshape shape(static_cast<index_t>(vec.size()));
  auto ptr = thrust::raw_pointer_cast(vec.data());
  return contiguous_view<T, sshape, device::cpu>(ptr, shape);
}

}  // namespace mathprim
