#pragma once
#include <mutex>  // IWYU pragma: export

#ifndef MATHPRIM_WARN_ONCE
#  define MATHPRIM_WARN_ONCE(msg)       \
    {                                   \
      static ::std::once_flag flag;     \
      ::std::call_once(flag, []() {     \
        fprintf(stderr, "%s\n", (msg)); \
      });                               \
    }
#endif

namespace mathprim::internal {

template <typename T = void>
static constexpr bool always_false_v = !std::is_same_v<T, T>;

}