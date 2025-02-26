#pragma once
#include "mathprim/core/view.hpp"
namespace mathprim::optim {

template <typename Scalar>
struct optim_result {
  Scalar value;
  int iterations;
};

template <typename Scalar, typename Sshape, typename Sstride, typename Device>
struct data_item {
  using scalar_type = Scalar;
  using device_type = Device;

  // A view to the data.
  basic_view<Scalar, Sshape, Sstride, Device> value_;

  // A view to the gradient.
  basic_view<Scalar, Sshape, Sstride, Device> gradient_;
};

template <typename Derived>
struct basic_optim_problem {};
}  // namespace mathprim::optim