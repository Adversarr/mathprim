#pragma once
#include <mutex>

namespace mathprim {
using std::call_once;

#ifndef MATHPRIM_WARN_ONCE
#  define MATHPRIM_WARN_ONCE(msg)       \
    {                                   \
      static std::once_flag flag;       \
      call_once(flag, []() {            \
        fprintf(stderr, "%s\n", (msg)); \
      });                               \
    }
#endif

}  // namespace mathprim
