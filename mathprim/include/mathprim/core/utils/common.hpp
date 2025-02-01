#pragma once
#include <mutex>  // IWYU pragma: export

#include "mathprim/core/defines.hpp"

#ifndef MATHPRIM_WARN_ONCE
#  define MATHPRIM_WARN_ONCE(msg)       \
    {                                   \
      static ::std::once_flag flag;     \
      ::std::call_once(flag, []() {     \
        fprintf(stderr, "%s\n", (msg)); \
      });                               \
    }
#endif

namespace mathprim {
namespace internal {

template <typename T = void>
static constexpr bool always_false_v = !std::is_same_v<T, T>;

template <template <typename> typename Base, typename T>
std::true_type is_base_of_template(const Base<T> *);

template <template <typename> typename Base>
std::false_type is_base_of_template(...);

template <typename T, template <typename> typename Base>
constexpr bool is_base_of_template_v = decltype(is_base_of_template<Base>(std::declval<T *>()))::value;
}  // namespace internal

MATHPRIM_PRIMFUNC index_t ceil_div(index_t a, index_t b) noexcept {
  return (a + b - 1) / b;
}

}  // namespace mathprim
