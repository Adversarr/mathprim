#pragma once
#include <omp.h>

#include "mathprim/core/defines.hpp"
#include "mathprim/parallel/parallel.hpp"  // IWYU pragma: export

namespace mathprim {

namespace par {

class openmp : public parfor<openmp> {
  index_t grain_size_ = 1 << 7;  ///< The grain size to use parallel for.
  index_t threshold_ = 1 << 14;  ///< The threshold to use parallel for.

public:
  openmp() noexcept = default;

  /**
   * @brief Set the grain size (min number of elements per thread).
   * 
   * @param grain_size 
   * @return openmp& 
   */
  openmp& set_grain_size(index_t grain_size) noexcept {
    grain_size_ = grain_size;
    return *this;
  }

  /**
   * @brief Set the threshold (min number of elements to parallelize).
   * 
   * @param threshold 
   * @return openmp& 
   */
  openmp& set_threshold(index_t threshold) noexcept {
    threshold_ = threshold;
    return *this;
  }

  /**
   * @brief Set the number of threads to use.
   * 
   * @param num_threads 
   * @return openmp& 
   */
  openmp& set_num_threads(index_t num_threads) noexcept {
    omp_set_num_threads(num_threads);
    return *this;
  }

  /**
   * @brief Get the number of threads.
   * 
   * @return int 
   */
  int get_num_threads() const noexcept {
    return omp_get_max_threads();
  }

  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run_impl(index_pack<sgrids...> grid_dim, index_pack<sblocks...> block_dim, Fn&& fn) const noexcept {
    const index_t total = grid_dim.numel();
    if (total < threshold_) {
      seq{}.run(grid_dim, block_dim, std::forward<Fn>(fn));
    } else {
      const index_t block_total = block_dim.numel();
#pragma omp parallel for schedule(static)
      for (index_t i = 0; i < total; ++i) {
        const auto grid_id = ind2sub(grid_dim, i);
        const auto block_dim_local = block_dim;

        for (index_t j = 0; j < block_total; ++j) {
          const auto block_id = ind2sub(block_dim_local, j);
          fn(grid_id, block_id);
        }
      }
    }
  }

  template <typename Fn, index_t... Sgrids>
  void run_impl(index_pack<Sgrids...> grid_dim, Fn&& fn) const noexcept {
    const index_t total = grid_dim.numel();
    if (total < threshold_) {
      seq{}.run(grid_dim, std::forward<Fn>(fn));
    } else {
      const index_t n = total / grain_size_;
      const int threads = get_num_threads();
      const index_t chunk_size = (n + threads - 1) / (threads);
      MATHPRIM_UNUSED(chunk_size);
#pragma omp parallel for schedule(static, chunk_size) 
      for (index_t block_id = 0; block_id < n; ++block_id) {
        // Individual task load = grain_size.
        const index_t beg = block_id * grain_size_;
        const index_t end = std::min(beg + grain_size_, total);
        const auto gd_local = grid_dim;

        for (index_t i = beg; i < end; ++i) {
          const auto grid_id = ind2sub(gd_local, i);
          fn(grid_id);
        }
      }
    }
  }
};

}  // namespace par
}  // namespace mathprim
