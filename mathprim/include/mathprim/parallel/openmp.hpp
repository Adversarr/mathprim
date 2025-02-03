#pragma once
#include "mathprim/core/defines.hpp"
#ifndef MATHPRIM_ENABLE_OPENMP
#  error "OpenMP is not enabled"
#endif

#include <omp.h>

#include "mathprim/parallel/parallel.hpp"  // IWYU pragma: export

namespace mathprim {

namespace par {

class openmp : public parfor<openmp> {
public:
  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run_impl(const index_pack<sgrids...>& grid_dim, const index_pack<sblocks...>& block_dim,
                Fn&& fn) const noexcept {
    const index_t total = grid_dim.numel();
    const index_t threads = static_cast<index_t>(omp_get_max_threads());
    const index_t chunk_size = total / threads;
#pragma omp parallel for schedule(static) firstprivate(fn, total, threads, grid_dim, block_dim, chunk_size)
    for (index_t i = 0; i < total; ++i) {
      auto grid_id = ind2sub(grid_dim, i);
      for (auto block_id : block_dim) {
        fn(grid_id, block_id);
      }
    }
  }

  template <typename Fn, index_t... sgrids>
  void run_impl(const index_pack<sgrids...>& grid_dim, Fn&& fn) const noexcept {
    const index_t total = grid_dim.numel();
    const index_t threads = static_cast<index_t>(omp_get_max_threads());
    const index_t chunk_size = total / threads;
#pragma omp parallel for schedule(static) firstprivate(fn, total, threads, grid_dim, chunk_size)
    for (index_t i = 0; i < total; ++i) {
      auto grid_id = ind2sub(grid_dim, i);
      fn(grid_id);
    }
  }
};

}  // namespace par
}  // namespace mathprim
