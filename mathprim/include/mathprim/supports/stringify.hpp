#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "mathprim/core/common.hpp"  // IWYU pragma: export
#include "mathprim/core/defines.hpp"

namespace mathprim {

template <index_t... svalues>
std::ostream& operator<<(std::ostream& os, const index_pack<svalues...>& pack) {
  os << "(";
  for (index_t i = 0; i < ndim(pack) - 1; ++i) {
    os << pack[i] << ", ";
  }
  os << pack.template get<-1>() << ")";
  return os;
}

template <index_t ndim>
std::ostream& operator<<(std::ostream& os, const index_array<ndim>& pack) {
  os << "(";
  for (index_t i = 0; i < ndim - 1; ++i) {
    os << pack[i] << ", ";
  }
  os << pack[ndim - 1] << ")";
  return os;
}

// template <typename T, index_t N, device_t dev>
// std::ostream& operator<<(std::ostream& os, const basic_buffer<T, N, dev>& buffer) {
//   os << "buffer(" << static_cast<const void*>(buffer.data());
//   os << ", shape=(";
//   for (index_t i = 0; i < N; ++i) {
//     os << buffer.shape()[i];
//     os << ", ";
//   }
//   os << "), stride=(";
//   for (index_t i = 0; i < N; ++i) {
//     os << buffer.stride()[i] << ", ";
//   }
//   os << "), device=" << buffer.device() << ")";
//   return os;
// }
//
// template <typename T, index_t N, device_t dev>
// std::ostream& operator<<(std::ostream& os, const basic_view<T, N, dev>& view) {
//   os << "view(" << static_cast<const void*>(view.data());
//   os << ", shape=(";
//   for (index_t i = 0; i < N; ++i) {
//     os << view.shape()[i] << ", ";
//   }
//   os << "), stride=(";
//   for (index_t i = 0; i < N; ++i) {
//     os << view.stride()[i] << ", ";
//   }
//   os << "), device=" << view.device() << ")";
//   return os;
// }

}  // namespace mathprim
