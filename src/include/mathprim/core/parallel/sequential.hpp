#pragma once

#include "mathprim/core/dim.hpp"  // IWYU pragma: export
#include "mathprim/core/parallel.hpp"

namespace mathprim {

template <> struct parallel_backend_impl<par::seq> {
  template <typename Fn, index_t N>
  static void foreach_index(const dim<N>& grid_dim, const dim<N>& block_dim,
                            Fn fn) {
    for (auto grid_id : grid_dim) {
      for (auto block_id : block_dim) {
        fn(grid_id, block_id);
      }
    }
  }
};

using parfor_seq = parfor<par::seq>;

}  // namespace mathprim
