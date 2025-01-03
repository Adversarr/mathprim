#pragma once
#include "parallel/sequential.hpp"

namespace mathprim::parallel {

template <parallel_t parallel>
struct foreach_index {
  using impl_type = parallel_backend_traits<parallel>;
  template <typename Fn>
  static void launch(const dim_t& grid_dim, const dim_t& block_dim, Fn fn) {
    impl_type::foreach_index(grid_dim, block_dim, fn);
  }

  template <typename Fn>
  static void launch(const dim_t& grid_dim, Fn fn) {
    launch(grid_dim, dim_t{1},
           [fn](const dim_t& grid_id, const dim_t& /* block_id */) {
             fn(grid_id);
           });
  }
};

}  // namespace mathprim::parallel