#pragma once
#include "buffers/basic_buffer.hpp"
#include "buffers/cpu_buffer.hpp"

namespace mathprim {

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T the data type of the buffer.
 * @param shape the shape of the buffer.
 * @return the buffer, throw exception if failed.
 */
template <typename T>
basic_buffer<T> make_buffer(const dim_t &shape) {
  return backend::cpu::make_buffer<T>(shape);
}

}  // namespace mathprim
