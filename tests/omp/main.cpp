#include <iostream>
#include <mathprim/core/parallel/openmp.hpp>
#include <mathprim/supports/stringify.hpp>
#include <mutex>
using namespace mathprim;

int main() {
  dim_t grid_dim{4, 3};
  dim_t block_dim{2, 2};

  std::mutex mtx;
  parallel::openmp::foreach_index(
      grid_dim, block_dim, [&mtx](dim_t grid_id, dim_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Grid ID: " << grid_id << ", Block ID: " << block_id
                  << "Thread ID: " << omp_get_thread_num() << std::endl;
      });
  return 0;
}