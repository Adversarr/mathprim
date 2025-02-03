#pragma once

#include "mathprim/core/defines.hpp"
#include "mathprim/parallel/parallel.hpp"

namespace mathprim {

namespace par {

class seq : public parfor<seq> {
public:
  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run_impl(const index_pack<sgrids...>& grid_dim, const index_pack<sblocks...>& block_dim,
                Fn&& fn) const noexcept {
    for (auto grid_id : grid_dim) {
      for (auto block_id : block_dim) {
        fn(grid_id, block_id);
      }
    }
  }

  template <typename Fn, index_t... sgrids>
  void run_impl(const index_pack<sgrids...>& grid_dim, Fn&& fn) const noexcept {
    for (auto grid_id : grid_dim) {
      fn(grid_id);
    }
  }
};
}  // namespace par

using parfor_seq = parfor<par::seq>;

}  // namespace mathprim
