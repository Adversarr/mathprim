#include "mathprim/sparse/gather.hpp"
#include <iostream>

using namespace mathprim;

int main() {
  // For example, I want a 1D laplacian matrix, built with Local->Global mapping.
  par::seq exec;
  index_t N = 4;

  using namespace literal;
  auto local_laplacians = make_buffer<float>(N - 1, 2_s, 2_s);

  std::vector<sparse::entry<double>> entries;
  for (index_t i = 0; i < N - 1; ++i) {
    // For each element, we have 2 nodes, and the local laplacian 
    index_t left = i;
    index_t right = i + 1;
  }
}