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
  index_t grain_size_ = 1 << 7;  ///< The grain size to use parallel for.
  index_t threshold_ = 1 << 14;  ///< The threshold to use parallel for.

public:
  openmp() noexcept = default;

  openmp& set_grain_size(index_t grain_size) noexcept {
    grain_size_ = grain_size;
    return *this;
  }

  openmp& set_threshold(index_t threshold) noexcept {
    threshold_ = threshold;
    return *this;
  }

  static void set_num_threads(index_t num_threads) noexcept {
    omp_set_num_threads(num_threads);
  }

  static int get_num_threads() noexcept {
    return omp_get_max_threads();
  }

  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run_impl(index_pack<sgrids...> grid_dim, index_pack<sblocks...> block_dim, Fn fn) const noexcept {
    const index_t total = grid_dim.numel();
    if (total < threshold_) {
      seq{}.run_impl(grid_dim, fn);
    } else {
#pragma omp parallel for schedule(static) firstprivate(fn, total, grid_dim)
      for (index_t i = 0; i < total; ++i) {
        const auto grid_id = ind2sub(grid_dim, i);

#pragma omp simd
        for (auto block_id : block_dim) {
          fn(grid_id, block_id);
        }
      }
    }
  }

  template <typename Fn, index_t... sgrids>
  void run_impl(index_pack<sgrids...> grid_dim, Fn fn) const noexcept {
    const index_t total = grid_dim.numel();
    if (total < threshold_) {
      seq{}.run_impl(grid_dim, fn);
    } else {
      const index_t n = total / grain_size_;
#pragma omp parallel for schedule(static) firstprivate(fn, total, grid_dim, grain_size_)
      for (index_t block_id = 0; block_id < n; ++block_id) {
        const index_t beg = block_id * grain_size_;
        const index_t end = std::min(beg + grain_size_, total);

#pragma omp simd
        for (index_t i = beg; i < end; ++i) {
          const auto grid_id = ind2sub(grid_dim, i);
          fn(grid_id);
        }
      }
    }
  }
};

}  // namespace par
}  // namespace mathprim
