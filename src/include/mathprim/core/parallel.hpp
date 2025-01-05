#pragma once
#include "mathprim/core/dim.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/parallel/std.hpp"
#include "mathprim/core/parallel/sequential.hpp"
namespace mathprim {

template <par parallel> struct parfor {
  using impl_type = parallel_backend_impl<parallel>;

  template <typename Fn>
  static void run(const dim_t& grid_dim, const dim_t& block_dim, Fn fn) {
    impl_type::foreach_index(grid_dim, block_dim, fn);
  }

  template <typename Fn> static void run(const dim_t& grid_dim, Fn fn) {
    run(grid_dim, dim_t{1},
           [fn](const dim_t& grid_id, const dim_t& /* block_id */) {
             fn(grid_id);
           });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each(basic_buffer_view<T, N, dev> buffer, Fn fn) {
    run(dim_t{buffer.shape()}, [fn, buffer](const dim_t& idx) {
      const dim<N> downgraded_idx{idx};
      fn(buffer(downgraded_idx));
    });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each_indexed(basic_buffer_view<T, N, dev> buffer, Fn fn) {
    run(dim_t{buffer.shape()}, [fn, buffer](const dim_t& idx) {
      const dim<N> downgraded_idx{idx};
      fn(downgraded_idx, buffer(downgraded_idx));
    });
  }
};

}  // namespace mathprim
