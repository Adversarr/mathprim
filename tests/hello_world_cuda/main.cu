#include <iostream>
#define MATHPRIM_VERBOSE_MALLOC 1

#include <mathprim/core/buffers/cuda_buffer.cuh>
#include <mathprim/core/common.hpp>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;

int main() {
  auto buffer = make_buffer<float, device_t::cuda>(1);
  auto [x, y] = buffer.shape().xy();
  std::cout << "Buffer shape: " << x << " " << y << std::endl;
  std::cout << "Buffer shape: " << buffer.shape() << std::endl;
  std::cout << "Buffer stride: " << buffer.stride() << std::endl;
  std::cout << "Buffer: " << buffer << std::endl;
  return 0;
}
