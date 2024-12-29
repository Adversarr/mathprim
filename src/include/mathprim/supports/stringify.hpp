#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "mathprim/core/common.hpp"

namespace mathprim {

template <index_t N>
std::ostream& operator<<(std::ostream& os, const dim<N>& dim) {
  os << "dim" << N << "(";
  const index_t ndim = dim.ndim();
  for (index_t i = 0; i < ndim; ++i) {
    os << dim[i];
    if (i < ndim - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const basic_buffer<T>& buffer) {
  os << "buffer(" << static_cast<const void*>(buffer.data());
  const index_t ndim = buffer.ndim();
  os << ", shape=(";
  for (index_t i = 0; i < ndim; ++i) {
    os << buffer.shape()[i];
    if (i < ndim - 1) {
      os << ", ";
    }
  }
  os << "), stride=(";
  for (index_t i = 0; i < ndim; ++i) {
    os << buffer.stride()[i];
    if (i < ndim - 1) {
      os << ", ";
    }
  }
  os << "))";
  return os;
}

template <typename T>
std::string to_string(const basic_buffer<T>& buffer) {
  std::ostringstream os;
  os << buffer;
  return os.str();
}

template <index_t N>
std::string to_string(const dim<N>& dim) {
  std::ostringstream os;
  os << dim;
  return os.str();
}
}  // namespace mathprim
