#include <cstring>
#include <iostream>
#include <mathprim/core/common.hpp>
#include <mathprim/core/view_from/stl_vector.hpp>
#include <mathprim/supports/stringify.hpp>

#include <mathprim/dynamic/buffer.hpp>

using namespace mathprim;

void test_convertible(const_f32_buffer_view_3d<device_t::cpu> v) {
  std::cout << "Convertible!" << v << std::endl;
}

void print_content(const_index_buffer_view_2d<device_t::cpu> v) {
  for (auto row: v) {
    for (auto item: row) {
      std::cout << item << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  auto buffer = make_buffer<float>(dim{2, 3, 4});
  std::cout << "Buffer shape: " << buffer.shape() << std::endl;
  std::cout << "Buffer stride: " << buffer.stride() << std::endl;
  std::cout << "Buffer: " << buffer << std::endl;

  auto view = buffer.view();
  std::cout << "View: " << view << std::endl;

  for (int i = 0; i < buffer.size(); ++i) {
    printf("buffer[%d] = %f\n", i, view[i]);
    view[i] = i;
  }

  auto buffer2 = make_buffer<float>(3, 4);

  auto view2 = buffer2.view();
  copy<float, 2, device_t::cpu>(view2, view.slice(0));

  for (auto [i, j] : buffer2.shape()) {
    printf("buffer2[%d, %d] = %f\n", i, j, view2(i, j));
  }

  test_convertible(view);
  {
    auto buf_p = dynamic::make_buffer_ptr<int>(dim(3, 4), device_t::cpu);
    auto p_view = buf_p->view();
    for (auto [i, j]: p_view.shape()) {
      p_view(i, j) = i + j;
    }
    print_content(p_view.as_const());
  }

  auto view3 = view2.view<2>({2, 6});
  std::cout << "View2: " << view2 << std::endl;
  std::cout << "View3: " << view3 << std::endl;
  for (auto [i, j]: view3.shape()) {
    printf("view3(%d, %d) = %f\n", i, j, view3(i, j));
  }

  return 0;
}
