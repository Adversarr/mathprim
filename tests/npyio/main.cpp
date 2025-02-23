#include <fstream>
#include <iostream>

#include "mathprim/supports/io/npy.hpp"
#include "mathprim/supports/stringify.hpp"

using namespace mathprim;

int main() {
  auto example = make_buffer<float>(2, 3, 4);
  for (index_t i = 0; i < 24; ++i) {
    example.data()[i] = static_cast<float>(i);
  }

  io::numpy<float, 3> writer;
  std::ofstream out("example.npy", std::ios_base::binary);
  writer.write(out, example.view());

  std::ifstream inp("example2.npy", std::ios_base::binary);
  auto buf = writer.read(inp);

  std::cout << "Read buffer: " << buf << std::endl;
  for (index_t i = 0; i < 24; ++i) {
    if (example.data()[i] != buf.data()[i]) {
      std::cerr << "Mismatch at " << i << ": " << example.data()[i] << " != " << buf.data()[i] << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}