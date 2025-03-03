#pragma once
#include <cmath>

#include "mathprim/dnn/basic_module.hpp"
namespace mathprim::dnn {

template <typename Scalar>
struct identity_activation {
  // x->fx
  MATHPRIM_PRIMFUNC Scalar fwd(Scalar x) const noexcept { return x; }
  // x, fx->f'(x)
  MATHPRIM_PRIMFUNC Scalar bwd(Scalar /* x */, Scalar /* fx */) const noexcept { return 1; }
};

template <typename Scalar>
struct sigmoid_activation {
  MATHPRIM_PRIMFUNC Scalar fwd(Scalar x) const noexcept { return 1 / (1 + ::exp(-x)); }

  MATHPRIM_PRIMFUNC Scalar bwd(Scalar /* x */, Scalar fx) const noexcept { return fx * (1 - fx); }
};

template <typename Scalar, typename Device, typename InShape, template <typename> typename Activation>
class activation;

template <typename Scalar, typename InShape, template <typename> typename Activation>
class activation<Scalar, device::cpu, InShape, Activation>
    : public basic_module<activation<Scalar, device::cpu, InShape, Activation>, Scalar, device::cpu, InShape, InShape> {
public:
  using base
      = basic_module<activation<Scalar, device::cpu, InShape, Activation>, Scalar, device::cpu, InShape, InShape>;
  friend base;
  using out_batch = typename base::out_batch;
  using const_in_batch = typename base::const_in_batch;
  using const_out_batch = typename base::const_out_batch;
  using compile_return_t = typename base::compile_return_t;
  template <typename Blas>
  using ctx = typename base::template ctx<Blas>;
  using in_shape = typename base::in_shape;
  using out_shape = typename base::out_shape;
  using out_buffer = to_buffer_t<out_batch>;

  activation() = default;
  explicit activation(InShape in_shape) : in_shape_(in_shape) {}
  MATHPRIM_INTERNAL_MOVE(activation, default);



private:
  out_buffer out_;
  InShape in_shape_;
};

}  // namespace mathprim::dnn