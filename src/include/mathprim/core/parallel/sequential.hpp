#pragma once

#include "mathprim/core/dim.hpp"

namespace mathprim {

namespace parallel::seq {

template <typename Fn>
void foreach_index(const dim_t& grid_dim, const dim_t& block_dim, Fn fn) {
  for (auto grid_id : grid_dim) {
    for (auto block_id : block_dim) {
      fn(grid_id, block_id);
    }
  }
}

}  // namespace parallel::seq

template <>
struct parallel_backend_traits<parallel_t::none> {
  template <typename Fn>
  static void foreach_index(const dim_t& grid_dim, const dim_t& block_dim,
                            Fn fn) {
    parallel::seq::foreach_index(grid_dim, block_dim, std::forward<Fn>(fn));
  }
};

}  // namespace mathprim
