#pragma once
#include <omp.h>

#include <thread>
#include <vector>

#include "mathprim/core/dim.hpp"
#include "mathprim/core/parallel.hpp" // IWYU pragma: export

namespace mathprim {

namespace par {
class stl {
public:
  // TODO: Use a thread pool to avoid creating threads every time.
  template <typename Fn, index_t N>
  static void foreach_index(const dim<N> &grid_dim, const dim<N> &block_dim,
                            Fn fn) {
    index_t total = grid_dim.numel();
    index_t max_threads = ::std::thread::hardware_concurrency();
    index_t chunk_size = (total + max_threads - 1) / max_threads;

    auto worker = [&](index_t start, index_t end) {
      for (index_t i = start; i < end; ++i) {
        const dim<N> sub_id = ind2sub(grid_dim, i);
        for (auto block_id : block_dim) {
          fn(sub_id, block_id);
        }
      }
    };

    std::vector<::std::thread> threads;
    for (index_t t = 0; t < max_threads; ++t) {
      index_t start = t * chunk_size;
      index_t end = ::std::min(start + chunk_size, total);
      threads.emplace_back(worker, start, end);
    }

    for (auto &thread : threads) {
      thread.join();
    }
  }
};
} // namespace par

} // namespace mathprim
