#pragma once

#include "mathprim/dnn/basic_module.hpp"

namespace mathprim::dnn {

/**
 * @brief It computes Y <- Reshape(X)
 *
 * @tparam Scalar
 * @tparam Device
 */
template <typename Scalar, typename Device, typename InShape, typename OutShape>
class reshape : public basic_module<reshape<Scalar, Device, InShape, OutShape>, Scalar, Device, InShape, OutShape> {
public:
  using base = basic_module<reshape<Scalar, Device, InShape, OutShape>, Scalar, Device, InShape, OutShape>;
  friend base;
  using in_batch = typename base::in_batch;
  using out_batch = typename base::out_batch;
  using const_in_batch = typename base::const_in_batch;
  using const_out_batch = typename base::const_out_batch;
  using compile_return_t = typename base::compile_return_t;
  template <typename Blas, typename ParImpl>
  using ctx_t = typename base::template ctx_t<Blas, ParImpl>;

  using in_shape = typename base::in_shape;
  using out_shape = typename base::out_shape;

  reshape(reshape&&) = default;
  explicit reshape(const InShape& in_shape, const OutShape& out_shape) : in_shape_(in_shape), out_shape_(out_shape) {
    MATHPRIM_INTERNAL_CHECK_THROW(in_shape.numel() == out_shape.numel(), std::invalid_argument,
                                  "The number of elements must be the same.");
  }

  template <typename Blas, typename ParImpl>
  compile_return_t compile_impl(ctx_t<Blas, ParImpl>& /* c */) {
    auto in_batch = base::input();
    auto batch_size = in_batch.shape(0);
    auto bshape = batched_shape(batch_size, out_shape_);
    auto y = base::input().reshape(bshape);
    auto dldy = base::curr_dl_dx_.reshape(bshape);
    return {y, dldy};
  }

  template <typename Blas, typename ParImpl>
  void zero_grad_impl(ctx_t<Blas, ParImpl>& /* c */) {}

  void reset_parameters_impl() {}

  template <typename Blas, typename ParImpl>
  out_batch forward_impl(ctx_t<Blas, ParImpl>& c) {}
  template <typename Blas, typename ParImpl>
  void backward_impl(ctx_t<Blas, ParImpl>& c, bool compute_dldx) {}

  in_shape input_shape_impl() const noexcept { return in_shape_; }
  out_shape out_shape_impl() const noexcept { return out_shape_; }

  InShape in_shape_;
  OutShape out_shape_;
};
}  // namespace mathprim::dnn