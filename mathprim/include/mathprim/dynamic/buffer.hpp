#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/backends/cpu.hpp"
#ifdef MATHPRIM_ENABLE_CUDA
#include "mathprim/core/backends/cuda.cuh"
#endif

namespace mathprim {

namespace dynamic {

// Create a buffer accordingly.
template <typename T, index_t N>
basic_buffer_ptr<T, N, device_t::dynamic> make_buffer_ptr(const dim<N>& shape, device_t device) {
  if (device == device_t::dynamic) {
    throw std::invalid_argument("dynamic buffer creation cannot be used without explicit device specification.");
  }

  if (device == device_t::cpu) {
    auto buffer = ::mathprim::make_buffer<T, N, device_t::cpu>(shape);
    return std::make_unique<basic_buffer<T, N, device_t::dynamic>>(std::move(buffer));
  } else if (device == device_t::cuda) {
    auto buffer = ::mathprim::make_buffer<T, N, device_t::cuda>(shape);
    return std::make_unique<basic_buffer<T, N, device_t::dynamic>>(std::move(buffer));
  } else {
    throw std::invalid_argument("invalid device value.");
  }
}

}
}