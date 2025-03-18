#include "checkings.hpp"

void add_1(nb::ndarray<int, nb::shape<-1>, nb::device::cpu, nb::numpy> arr) {
  auto cnt = arr.size();
  auto* ptr = arr.data();
  for (ssize_t i = 0; i < cnt; ++i) {
    ptr[i] += 1;
  }
}

void magic_number(int magic) {
  if (magic != 3407) {
    throw std::runtime_error("No Magic");
  }
}

void bind_checkings(nb::module_& m) {
  m.def("add_1", add_1, "Add 1 to each element of the input array.", nb::arg("arr"));
  m.def("magic_number", magic_number, "Check if the magic number is 3407.", nb::arg("magic"));
}
