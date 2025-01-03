#pragma once

#include "mathprim/core/dim.hpp"

namespace mathprim {

namespace parallel::cuda {
template <typename Fn>
__global__ void do_work(Fn fn, dim_t grid_dim, dim_t block_dim) {
  dim_t grid_id = {static_cast<index_t>(blockIdx.x),
                   static_cast<index_t>(blockIdx.y),
                   static_cast<index_t>(blockIdx.z)};
  dim_t block_id = {static_cast<index_t>(threadIdx.x),
                    static_cast<index_t>(threadIdx.y),
                    static_cast<index_t>(threadIdx.z)};

  for (index_t grid_w = 0; grid_w < internal::to_valid_size(grid_dim.w_);
       ++grid_w) {
    for (index_t block_w = 0; block_w < internal::to_valid_size(block_dim.w_);
         ++block_w) {
      grid_id.w_ = grid_w;
      block_id.w_ = block_w;

      fn(grid_id, block_id);
    }
  }
}

template <typename Fn>
void foreach_index(const dim_t &grid_dim, const dim_t &block_dim, Fn fn) {
  dim3 grid{static_cast<unsigned int>(internal::to_valid_size(grid_dim.x_)),
            static_cast<unsigned int>(internal::to_valid_size(grid_dim.y_)),
            static_cast<unsigned int>(internal::to_valid_size(grid_dim.z_))};
  dim3 block{static_cast<unsigned int>(internal::to_valid_size(block_dim.x_)),
             static_cast<unsigned int>(internal::to_valid_size(block_dim.y_)),
             static_cast<unsigned int>(internal::to_valid_size(block_dim.z_))};
  do_work<<<grid, block>>>(fn, grid_dim, block_dim);
}

} // namespace parallel::cuda

template <> struct parallel_backend_traits<parallel_t::cuda> {
  template <typename Fn>
  static void foreach_index(const dim_t &grid_dim, const dim_t &block_dim,
                            Fn fn) {
    parallel::cuda::foreach_index(grid_dim, block_dim, fn);
  }
};

template <> struct foreach_index<parallel_t::cuda> {
  using impl_type = parallel_backend_traits<parallel_t::cuda>;
  template <typename Fn>
  static void launch(const dim_t &grid_dim, const dim_t &block_dim, Fn fn) {
    impl_type::foreach_index(grid_dim, block_dim, fn);
  }

  template <typename Fn> static void launch(const dim_t &grid_dim, Fn fn) {
    launch(
        grid_dim, dim_t{1},
        [fn] MATHPRIM_GENERAL(const dim_t &grid_id,
                              const dim_t & /* block_id */) { fn(grid_id); });
  }
};

} // namespace mathprim
