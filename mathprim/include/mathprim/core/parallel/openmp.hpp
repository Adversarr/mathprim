#pragma once
#include "mathprim/core/defines.hpp"
#ifndef MATHPRIM_ENABLE_OPENMP
#error "OpenMP is not enabled"
#endif

#include <omp.h>
#include <type_traits>

#include "mathprim/core/dim.hpp"
#include "mathprim/core/parallel.hpp" // IWYU pragma: export

namespace mathprim {

namespace par {
class openmp {
public:
  template <typename Fn, index_t N>
  static void foreach_index(const dim<N> &grid_dim, const dim<N> &block_dim,
                            Fn fn) {
    const index_t total = grid_dim.numel();
    const index_t threads = static_cast<index_t>(omp_get_max_threads());
    const index_t chunk_size = total / threads;
    if constexpr (N > 1) {
#pragma omp parallel for schedule(static)                                      \
    firstprivate(fn, total, threads, grid_dim, block_dim, chunk_size)
      for (index_t i = 0; i < total; ++i) {
        const dim<N> block_id = ind2sub(grid_dim, i);
        for (const dim<N> thread_id : block_dim) {
          fn(block_id, thread_id);
        }
      }
    } else {
      index_t numel_in_block = block_dim.numel();
#pragma omp parallel for schedule(static)                                      \
    firstprivate(fn, total, threads, numel_in_block, chunk_size)
      for (index_t i = 0; i < total; ++i) {
        for (index_t j = 0; j < numel_in_block; ++j) {
          fn(dim<1>{i}, dim<1>{j});
        }
      }
    }
  }
};
} // namespace par
using parfor_openmp = parfor<par::openmp>;

} // namespace mathprim
