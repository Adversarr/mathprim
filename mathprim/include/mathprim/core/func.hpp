#pragma once
#include "mathprim/core/defines.hpp"
namespace mathprim {

template <typename Integer,
          typename = std::enable_if_t<std::is_integral_v<Integer>>>
MATHPRIM_PRIMFUNC Integer up_div(Integer a, Integer b) noexcept {
  return (a + b - 1) / b;
}

}  // namespace mathprim
