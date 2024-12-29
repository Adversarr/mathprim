#include <iostream>
#include <mathprim/core/common.hpp>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;

int main(int argc, char *argv[]) {
  auto buffer = make_buffer<float>(dim_t{1});
  std::cout << "Buffer shape: " << buffer.shape() << std::endl;
  std::cout << "Buffer stride: " << buffer.stride() << std::endl;
  std::cout << "Buffer: " << buffer << std::endl;
  return 0;
}
