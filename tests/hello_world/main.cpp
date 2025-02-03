#include <mathprim/core/buffer.hpp>
#include <mathprim/core/common.hpp>

using namespace mathprim;

int main() {
  float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto v = make_view(buf, shape_t<4>());
  // const view:
  basic_view<const float, shape_t<4>, stride_t<4>, device::cpu> v2 = v;
  // Should fail:
  // basic_view<const float, shape_t<4>, stride_t<2>, device::cpu> v3 = v;
  // basic_view<const float, shape_t<2>, stride_t<4>, device::cpu> v3 = v;
  basic_view<const float, shape_t<-1>, stride_t<4>, device::cpu> dyn_1 = v;
  // Should fail:
  // basic_view<float, shape_t<-1>, stride_t<4>, device::cpu> non_const = dyn_1;
}
