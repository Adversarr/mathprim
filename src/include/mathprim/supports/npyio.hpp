#pragma once

#include <fstream>

#include "mathprim/core/defines.hpp"
namespace mathprim::numpy {

// TODO: Implementation
template <typename T, index_t N>
void write(std::ofstream& file,
           const basic_buffer<T, N, device_t::cpu>& buffer);

// TODO: Implementation
template <typename T>
basic_buffer<T, max_ndim, device_t::cpu> read(std::ifstream& file);

}  // namespace mathprim::numpy