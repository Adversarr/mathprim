#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <iostream>
#include <mathprim/core/view.hpp>
#include <mathprim/supports/stringify.hpp>

int add(int a, int b) { return a + b; }

namespace nb = nanobind;
namespace mp = mathprim;

void print_mat_view(nb::ndarray<float, nb::shape<-1, -1>> & matrix) {
  float* data = matrix.data();
  auto view = mp::make_view(data, mp::shape_t<-1, -1>(matrix.shape(0), matrix.shape(1)),
                                  mp::stride_t<-1, -1>(matrix.stride(0) * 4, matrix.stride(1) * 4));
  std::cout << view << std::endl;
}

NB_MODULE(pymathprim, m) {
  m.def("add", &add);
  m.def("print_mat_view", &print_mat_view);
}