#include <cstring>
#include <iostream>
#include <mathprim/core/common.hpp>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;

void test_convertible(const_f32_buffer_view_3d<device_t::cpu> v) {
  std::cout << "Convertible!" << v << std::endl;
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

  for (auto [i, j]: buffer2.shape()) {
    printf("buffer2[%d, %d] = %f\n", i, j, view2(i, j));
  }

  test_convertible(view);
  return 0;
}
