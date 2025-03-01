#include "bind_cuda.cuh"
#include "mathprim/parallel/cuda.cuh"
#include "mathprim/supports/binding/nb_ext.hpp"
#include <iostream>

template <typename T, typename shape, typename dev>
static void test_view_nb2(nb::ndarray<T, shape, dev> arr) {
  auto view = nbex::to_mp_view_standard(arr);
  mathprim::par::cuda par;
  par.run(view.shape(), [view] __device__(const auto &idx) {
    printf("view(%d, %d) = %f\n", idx[0], idx[1], view(idx));
    view(idx) = 1;
  });
}

void do_binding_cuda(nb::module_ &m) {
  m.def("test_view_cu",
        &test_view_nb2<float, nb::shape<-1, -1>, nb::device::cuda>);
}