#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "mathprim/core/common.hpp"  // IWYU pragma: export
#include "mathprim/core/defines.hpp"

namespace mathprim {

inline std::ostream& operator<<(std::ostream& os, const device_t& device) {
  switch (device) {
    case device_t::cpu:
      os << "cpu";
      break;
    case device_t::cuda:
      os << "cuda";
      break;
    case device_t::dynamic:
      os << "dynamic";
      break;
    default:
      os << "Unknown";
      break;
  }
  return os;
}

template <index_t N>
std::ostream& operator<<(std::ostream& os, const dim<N>& dim) {
  os << "dim" << N << "(";
  for (index_t i = 0; i < N; ++i) {
    os << dim[i] << ", ";
  }
  os << ")";
  return os;
}

template <typename T, index_t N, device_t dev>
std::ostream& operator<<(std::ostream& os, const basic_buffer<T, N, dev>& buffer) {
  os << "buffer(" << static_cast<const void*>(buffer.data());
  os << ", shape=(";
  for (index_t i = 0; i < N; ++i) {
    os << buffer.shape()[i];
    os << ", ";
  }
  os << "), stride=(";
  for (index_t i = 0; i < N; ++i) {
    os << buffer.stride()[i] << ", ";
  }
  os << "), device=" << buffer.device() << ")";
  return os;
}

template <typename T, index_t N, device_t dev>
std::ostream& operator<<(std::ostream& os, const basic_view<T, N, dev>& view) {
  os << "view(" << static_cast<const void*>(view.data());
  os << ", shape=(";
  for (index_t i = 0; i < N; ++i) {
    os << view.shape()[i] << ", ";
  }
  os << "), stride=(";
  for (index_t i = 0; i < N; ++i) {
    os << view.stride()[i] << ", ";
  }
  os << "), device=" << view.device() << ")";
  return os;
}

template <typename T, index_t N, device_t dev>
std::string to_string(const basic_buffer<T, N, dev>& buffer) {
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

template <typename T, index_t N, device_t dev>
std::string to_string(const basic_view<T, N, dev>& view) {
  std::ostringstream os;
  os << view;
  return os.str();
}

}  // namespace mathprim
