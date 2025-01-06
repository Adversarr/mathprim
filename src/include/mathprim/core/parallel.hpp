#pragma once
#include "mathprim/core/dim.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/parallel/std.hpp"
#include "mathprim/core/parallel/sequential.hpp"
namespace mathprim {

template <par parallel> struct parfor {
  using impl_type = parallel_backend_impl<parallel>;

  template <typename Fn, index_t N>
  static void run(const dim<N>& grid_dim, const dim<N>& block_dim, Fn fn) {
    impl_type::template foreach_index<Fn, N>(grid_dim, block_dim, fn);
  }

  template <typename Fn, index_t N> static void run(const dim<N>& grid_dim, Fn fn) {
    run(grid_dim, dim<N>{1},
           [fn](const dim<N>& grid_id, const dim<N>& /* block_id */) {
             fn(grid_id);
           });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each(basic_buffer_view<T, N, dev> buffer, Fn fn) {
    run(buffer.shape(), [fn, buffer](const dim<N>& idx) {
      fn(buffer(idx));
    });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each_indexed(basic_buffer_view<T, N, dev> buffer, Fn fn) {
    parfor::run(buffer.shape(), [fn, buffer](const dim<N>& idx) {
      fn(idx, buffer(idx));
    });
  }
};

}  // namespace mathprim
