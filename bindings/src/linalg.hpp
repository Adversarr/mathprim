#pragma once
#include <chrono>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <nanobind/stl/tuple.h>
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

#include "common.hpp"

#define TOSTR(x) #x
void bind_linalg(nb::module_& m);
void bind_linalg_cuda(nb::module_& m);

namespace helper {
inline std::chrono::high_resolution_clock::time_point time_now() {
  return std::chrono::high_resolution_clock::now();
}

// Return the elapsed time in seconds
inline double time_elapsed(std::chrono::high_resolution_clock::time_point start,
                           std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(end - start).count();
}

inline double time_elapsed(std::chrono::high_resolution_clock::time_point start) {
  return time_elapsed(start, time_now());
}
}  // namespace helper
