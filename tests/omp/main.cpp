#include <iostream>
#include <mathprim/parallel/openmp.hpp>
#include <mathprim/supports/stringify.hpp>
#include <mutex>

#include "mathprim/core/buffer.hpp"
using namespace mathprim;

int main() {
  dshape<2> grid_dim{4, 3};
  dshape<2> block_dim{2, 2};

  std::mutex mtx;
  par::openmp{}.run(grid_dim, block_dim, [&mtx](auto grid_id, auto block_id) {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "Grid ID: " << grid_id << ", Block ID: " << block_id << " => Thread ID: " << omp_get_thread_num()
              << std::endl;
  });

  // start with a buffer.
  auto buf = make_buffer<float>(16);  // 16 elements.
  auto view = buf.view();
  par::openmp().run(view.shape(), [view](index_t i) {
    view[i] = i;
  });
  auto buf2 = make_buffer<float>(16, 4);
  auto view2 = buf2.view();
  par::openmp().run(view2.shape(), [view2](const auto &idx) {
    view2(idx) = idx[0];
  });

  par::openmp().vmap(
      [&mtx](auto a, auto b) {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << a << ": ";
        std::cout << b[0] << " " << b[1] << std::endl;
      },
      view, view2);

  return 0;
}
