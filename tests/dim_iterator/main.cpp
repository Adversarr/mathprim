#include <iostream>
#include <mathprim/core/dim.hpp>
#include <mathprim/supports/stringify.hpp>

using namespace mathprim;

int main() {
  dim_t start = {0, 0, 0};
  dim_t end = {2, 3, 4};
  auto it = dim_iterator<4>(start, end);
  auto ending = dim_iterator<4>({2, 0, 0}, end);
  for (auto i = it; i != ending; ++i) {
    static int cnt = 0;
    auto [x, y, z, w] = *i;
    std::cout << cnt << ": " << x << ", " << y << ", " << z << ", " << w << std::endl;
    if (++cnt == 100) {
      break;
    }
  }

  for (auto [x, y, z, w] : end) {
    std::cout << x << ", " << y << ", " << z << ", " << w << std::endl;
  }

  return 0;
}
