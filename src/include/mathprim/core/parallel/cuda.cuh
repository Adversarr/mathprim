#pragma once

#include "mathprim/core/dim.hpp"
#include "mathprim/core/parallel.hpp"

namespace mathprim {

namespace parallel::cuda {
template <typename Fn>
__global__ void do_work(Fn fn, dim_t grid_dim, dim_t block_dim) {
  dim_t grid_id
      = {static_cast<index_t>(blockIdx.x), static_cast<index_t>(blockIdx.y),
         static_cast<index_t>(blockIdx.z)};
  dim_t block_id
      = {static_cast<index_t>(threadIdx.x), static_cast<index_t>(threadIdx.y),
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

}  // namespace parallel::cuda

template <> struct parallel_backend_impl<par::cuda> {
  template <typename Fn>
  static void foreach_index(const dim_t &grid_dim, const dim_t &block_dim,
                            Fn fn) {
    parallel::cuda::foreach_index(grid_dim, block_dim, fn);
  }
};

template <> struct parfor<par::cuda> {
  using impl_type = parallel_backend_impl<par::cuda>;
  template <typename Fn>
  static void run(const dim_t &grid_dim, const dim_t &block_dim, Fn fn) {
    impl_type::foreach_index(grid_dim, block_dim, fn);
  }

  template <typename Fn> static void run(const dim_t &grid_dim, Fn fn) {
    run(grid_dim, dim_t{1},
        [fn] MATHPRIM_DEVICE(const dim_t &grid_id,
                             const dim_t & /* block_id */) {
          fn(grid_id);
        });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each(basic_buffer_view<T, N, dev> buffer, Fn fn) {
    run(dim_t{buffer.shape()}, [fn, buffer] MATHPRIM_DEVICE(const dim_t &idx) {
      const dim<N> downgraded_idx{idx};
      fn(buffer(downgraded_idx));
    });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each_indexed(basic_buffer_view<T, N, dev> buffer, Fn fn) {
    run(dim_t{buffer.shape()}, [fn, buffer] MATHPRIM_DEVICE(const dim_t &idx) {
      const dim<N> downgraded_idx{idx};
      fn(downgraded_idx, buffer(downgraded_idx));
    });
  }

  // we do not support for_each with multiple buffers, since we cannot guarantee
  // each buffer has the same shape
};

using parfor_cuda = parfor<par::cuda>;  ///< Alias for parfor<par::cuda>

}  // namespace mathprim
