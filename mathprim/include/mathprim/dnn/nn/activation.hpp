#pragma once
#include <cmath>

#include "mathprim/dnn/basic_module.hpp"
namespace mathprim::dnn {

template <typename Scalar>
struct identity_activation {
  // x->y=f(x)
  static MATHPRIM_PRIMFUNC Scalar fwd(Scalar x) noexcept { return x; }
  // x, dl/dy-> dl/dx
  static MATHPRIM_PRIMFUNC Scalar bwd(Scalar /* x */, Scalar dl_dy) noexcept { return dl_dy; }
};

template <typename Scalar>
struct relu_activation {
  static MATHPRIM_PRIMFUNC Scalar fwd(Scalar x) noexcept { return ::fmax(0, x); }
  static MATHPRIM_PRIMFUNC Scalar bwd(Scalar x, Scalar dl_dy) noexcept { return x >= 0 ? dl_dy : 0; }
};

template <typename Scalar>
struct sigmoid_activation {
  static MATHPRIM_PRIMFUNC Scalar fwd(Scalar x) noexcept { return 1 / (1 + std::exp(-x)); }
  static MATHPRIM_PRIMFUNC Scalar bwd(Scalar x, Scalar dl_dy) noexcept { 
    Scalar exp_x = std::exp(-x);
    return exp_x / ((1 + exp_x) * (1 + exp_x)) * dl_dy;
  }
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
  using in_batch = typename base::in_batch;
  using out_batch = typename base::out_batch;
  using const_in_batch = typename base::const_in_batch;
  using const_out_batch = typename base::const_out_batch;
  using compile_return_t = typename base::compile_return_t;
  template <typename Blas, typename ParImpl>
  using ctx_t = typename base::template ctx_t<Blas, ParImpl>;
  using in_shape = typename base::in_shape;
  using out_shape = typename base::out_shape;
  using out_buffer = to_buffer_t<out_batch>;

  activation() = default;
  explicit activation(InShape in_shape) : in_shape_(in_shape) {}

  MATHPRIM_INTERNAL_MOVE(activation, default);

  template <typename Blas, typename ParImpl>
  compile_return_t compile_impl(ctx_t<Blas, ParImpl>& /* c */) {
    MATHPRIM_ASSERT(in_shape_.numel() > 0);
    out_ = make_buffer<Scalar>(batched_shape(base::curr_batch_size_, in_shape_));
    out_grad_ = make_buffer<Scalar>(out_.shape());
    return {out_.view(), out_grad_.view()};
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Computes
  ///////////////////////////////////////////////////////////////////////////////

  template <typename Blas, typename ParImpl>
  void zero_grad_impl(ctx_t<Blas, ParImpl>& /* c */) {
    zeros(out_grad_);
  }

  template<typename Blas, typename ParImpl>
  out_batch forward_impl(ctx_t<Blas, ParImpl>& c) {
    auto x = base::curr_x_;
    auto y = out_.view();
    c.parallel().run(x.shape(), [x, y](auto i) {
      y(i) = Activation<Scalar>::fwd(x(i));
    });
    return y;
  }

  template <typename Blas, typename ParImpl>
  void backward_impl(ctx_t<Blas, ParImpl>& c, in_batch dl_dx) {
    if (!dl_dx) {
      return;
    }
    auto x = base::curr_x_;
    auto dl_dy = base::curr_dl_dy_;
    c.parallel().run(dl_dx.shape(), [x, dl_dx, dl_dy](auto i) {
      dl_dx(i) += Activation<Scalar>::bwd(x(i), dl_dy(i));
    });
  }

  ///////////////////////////////////////////////////////////////////////////////
  /// Meta data
  ///////////////////////////////////////////////////////////////////////////////
  in_shape input_shape_impl() const noexcept { return in_shape_; }
  out_shape out_shape_impl() const noexcept { return in_shape_; }
  index_t total_weights_impl() { return 0; }

private:
  out_buffer out_;
  out_buffer out_grad_;
  InShape in_shape_;
};

}  // namespace mathprim::dnn