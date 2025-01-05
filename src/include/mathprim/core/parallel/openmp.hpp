#pragma once
#ifndef MATHPRIM_ENABLE_OPENMP
#  error "OpenMP is not enabled"
#endif

#include <omp.h>

#include "mathprim/core/dim.hpp"
#include "mathprim/core/parallel.hpp"  // IWYU pragma: export

namespace mathprim {

template <> struct parallel_backend_impl<par::openmp> {
  template <typename Fn>
  static void foreach_index(const dim_t& grid_dim, const dim_t& block_dim,
                            Fn fn) {
    index_t total = grid_dim.numel();
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < total; ++i) {
      dim_t sub_id = ind2sub(grid_dim, i);
      for (auto block_id : block_dim) {
        fn(sub_id, block_id);
      }
    }
  }
};

using parfor_openmp = parfor<par::openmp>;

}  // namespace mathprim
