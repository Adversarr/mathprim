#pragma once
#include <omp.h>

#include <vector>

#include "mathprim/core/dim.hpp"
#include "mathprim/core/itertools/ndrange.hpp"

namespace mathprim {

namespace parallel::openmp {

template <typename Value, typename ReduceFn, typename TransformFn, typename... Args>
Value transpose_reduce(const dim_t& grid_dim, ReduceFn&& reduce_fn, TransformFn&& transform_fn,
                       Args&&... args) {
  index_t total = grid_dim.numel();
  std::vector<Value> results(total);  // zero-initialized

#pragma omp parallel
  {
    index_t tid = static_cast<index_t>(omp_get_thread_num());
    index_t num_threads = static_cast<index_t>(omp_get_num_threads());
    index_t chunk_size = (total + num_threads - 1) / num_threads;
    index_t start = tid * chunk_size;
    index_t end = std::min(start + chunk_size, total);

    for (index_t i = start; i < end; ++i) {
      dim_t sub_id = ind2sub(grid_dim, i);
      results[i] = reduce_fn(transform_fn(args...), results[i]);
    }
  }

  Value result = results[0];
  for (index_t i = 1; i < total; ++i) {
    result = reduce_fn(results[i], result);
  }
  return result;
}

template <typename Fn, typename... Args>
void foreach_index(const dim_t& grid_dim, Fn&& fn, Args&&... args) {
  index_t total = grid_dim.numel();
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < total; ++i) {
    dim_t sub_id = ind2sub(grid_dim, i);
    fn(sub_id, args...);
  }
}

}  // namespace parallel::openmp

template <>
struct parallel_backend_traits<parallel_t::openmp, device_t::cpu> {
  template <typename Fn, typename... Args>
  static void foreach_index(const dim_t& grid_dim, Fn&& fn, Args&&... args) {
    parallel::openmp::foreach_index(grid_dim, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }
};

}  // namespace mathprim
