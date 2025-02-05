
#include <iostream>
#include <mathprim/core/view.hpp>
#include <mathprim/supports/stringify.hpp>
#include <mathprim/core/devices/cuda.cuh>

#include "nb_ext.hpp"

int add(int a, int b) {
  return a + b;
}

namespace nb = nanobind;
namespace mp = mathprim;

void print_mat_view(nb::ndarray<float, nb::shape<-1, -1>>& matrix) {
  float* data = matrix.data();
  auto view = mp::make_view(data, mp::shape_t<-1, -1>(matrix.shape(0), matrix.shape(1)),
                            mp::stride_t<-1, -1>(matrix.stride(0) * 4, matrix.stride(1) * 4));
  std::cout << view << std::endl;
}

template <typename T, typename sshape, typename dev, typename sstride = mp::internal::default_stride_t<T, sshape>>
auto test_view() {
  static T* data = nullptr;
  mp::basic_view<T, sshape, sstride, dev> view(data, sshape());
  return nbex::to_nb_array_standard(view);
}

template <typename T, typename shape, typename dev>
void test_view_nb(nb::ndarray<T, shape, dev> arr) {
  auto view = nbex::to_mp_view_standard(arr);
  std::cout << view << std::endl;
  if constexpr (decltype(view)::ndim == 2 && std::is_same_v<dev, nb::device::cpu>) {
    for (int i = 0; i < view.shape(0); i++) {
      for (int j = 0; j < view.shape(1); j++) {
        std::cout << view(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }
}

auto test_field_f32x3() {
  mp::field_t<mathprim::cpu_vec3f32_const_view_t> view(nullptr, mp::shape_t<-1, 3>(4, 3));
  return nbex::to_nb_array_standard(view);
}

NB_MODULE(pymathprim, m) {
  m.def("add", &add);
  m.def("print_mat_view", &print_mat_view);
  m.def("test_view", &test_view<float, mp::shape_t<4>, mp::device::cpu>, nb::rv_policy::reference);
  m.def("test_view_nb", &test_view_nb<float, nb::shape<-1, 3>, nb::device::cpu>);
  m.def("test_view_nb", &test_view_nb<float, nb::shape<-1, -1>, nb::device::cpu>);
  m.def("test_view_cu", &test_view_nb<float, nb::shape<-1, -1>, nb::device::cuda>);
  m.def("test_field_f32x3", &test_field_f32x3, nb::rv_policy::reference);
}
