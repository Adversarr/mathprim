#pragma once

#include "mathprim/core/dim.hpp"

namespace mathprim {

namespace parallel::openmp {

template <typename Fn>
void foreach_index(const dim_t& grid_dim, const dim_t& block_dim, Fn&& fn) {
  for (auto grid_id : grid_dim) {
    for (auto block_id : block_dim) {
      fn(grid_id, block_id);
    }
  }
}

}  // namespace parallel::openmp

template <>
struct parallel_backend_traits<parallel_t::none, device_t::cpu> {
  template <typename Fn>
  static void foreach_index(const dim_t& grid_dim, const dim_t& block_dim,
                            Fn&& fn) {
    parallel::openmp::foreach_index(grid_dim, block_dim, std::forward<Fn>(fn));
  }
};

}  // namespace mathprim
