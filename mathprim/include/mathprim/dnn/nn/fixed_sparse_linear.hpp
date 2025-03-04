#pragma once
#include "mathprim/dnn/basic_module.hpp"
namespace mathprim::dnn {

template <typename Scalar, typename Device, typename SpBlas, typename InShape>
class fixed_sparse_linear
    : public basic_module<fixed_sparse_linear<Scalar, Device, SpBlas, InShape>, Scalar, Device, InShape, InShape> {
public:
  using base = basic_module<fixed_sparse_linear<Scalar, Device, SpBlas, InShape>, Scalar, Device, InShape, InShape>;
  friend base;
  explicit fixed_sparse_linear(SpBlas sparse);

private:

};
}