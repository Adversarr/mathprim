#pragma once

#include <ostream>
#include "mathprim/core/view.hpp"

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

template <typename T, typename sshape, typename sstride, typename dev>
std::ostream& operator<<(std::ostream& os, const basic_view<T, sshape, sstride, dev>& view) {
  if (view.is_contiguous()) {
    os << "view<data=" << view.data() << ", shape=" << view.shape() << ", sshape=" << sshape()
       << ", dev=" << dev{}.name() << ">";
  } else {
    os << "view<data=" << view.data() << ", shape=" << view.shape() << ", sshape=" << sshape()
       << ", stride=" << view.stride() << ", dev=" << dev{}.name() << ">";
  }
  return os;
}

}  // namespace mathprim
