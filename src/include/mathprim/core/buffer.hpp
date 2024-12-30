#pragma once
#include <type_traits>

#include "buffers/basic_buffer.hpp"
#include "mathprim/core/defines.hpp"

namespace mathprim {

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer(const dim_t &shape) {
  return buffer_traits<T, dev>::make_buffer(shape);
}

/**
 * @brief Alias of make_buffer.
 *
 */
template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer(index_t x) {
  return make_buffer<T, dev>(dim_t{x});
}

template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer_ptr(const dim_t &shape) {
  return std::make_unique<basic_buffer<T>>(make_buffer<T, dev>(shape));
}

template <typename T, device_t dev = device_t::cpu>
basic_buffer<T> make_buffer_ptr(index_t x) {
  return make_buffer_ptr<T, dev>(dim_t{x});
}

}  // namespace mathprim
