#include <iostream>
#include <mathprim/core/parallel.hpp>
#include <mathprim/core/parallel/openmp.hpp>
#include <mathprim/supports/stringify.hpp>
#include <mutex>
using namespace mathprim;

int main() {
  dim_t grid_dim{4, 3};
  dim_t block_dim{2, 2};

  std::mutex mtx;
  parfor<par::openmp>::run(
      grid_dim, block_dim, [&mtx](dim_t grid_id, dim_t block_id) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Grid ID: " << grid_id << ", Block ID: " << block_id
                  << "Thread ID: " << omp_get_thread_num() << std::endl;
      });

  // start with a buffer.
  auto buf = make_buffer<float>(16);  // 16 elements.
  parfor<par::openmp>::for_each_indexed(buf.view(),
                                        [](const dim<1>& idx, float& val) {
                                          val = idx[0];
                                        });

  auto bv = buf.view();
  for (float& i : bv) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  auto buf2 = make_buffer<float>(dim<2>{4, 4});

  parfor<par::openmp>::for_each_indexed(buf2.view(),
                                        [](const dim<2>& idx, float& val) {
                                          val = idx[0] * 4 + idx[1];
                                        });

  for (auto row : buf2.view()) {
    for (float i : row) {
      std::cout << i << " ";
    }
    std::cout << row << std::endl;
  }

  return 0;
}
