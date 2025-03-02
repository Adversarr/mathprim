#pragma once
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/supports/eigen_dense.hpp"
namespace mathprim::optim::ex_probs {

template <typename Scalar, typename Device>
class banana_problem;

template <typename Scalar>
class banana_problem<Scalar, device::cpu>
    : public basic_problem<banana_problem<Scalar, device::cpu>, Scalar, device::cpu> {
public:
  using base = basic_problem<banana_problem<Scalar, device::cpu>, Scalar, device::cpu>;
  friend base;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using buffer_type = typename base::buffer_type;
  using parameter = typename base::parameter;
  explicit banana_problem(index_t dsize, Scalar difficulty = 100) : difficulty_(difficulty) {
    x_ = make_buffer<Scalar>(dsize);
    base::register_parameter(x_.view());
  }

  banana_problem(banana_problem&&) noexcept = default;

  index_t eval_cnt() const { return cnt_; }

protected:
  void eval_value_and_gradients_impl() {
    auto x = eigen_support::cmap(x_.view());
    auto grad = eigen_support::cmap(base::at(0).gradient());
    grad.setZero();
    auto dsize = x_.shape(0);
    for (index_t i = 0; i < dsize - 1; i++) {
      Scalar nonlinear = (x(i) * x(i) - x(i + 1));
      base::accumulate_loss(nonlinear * nonlinear * difficulty_);
      Scalar linear = 1 - x(i);
      base::accumulate_loss(linear * linear);

      grad(i) += 4 * difficulty_ * nonlinear * x(i) - 2 * linear;
      grad(i + 1) += -2 * difficulty_ * nonlinear;
    }
    cnt_ += 1;
  }

  void eval_value_impl() { eval_value_and_gradients_impl(); }
  void eval_gradients_impl() { eval_value_and_gradients_impl(); }
  void on_setup() {
    // start from zero?
    x_.fill_bytes(0);
    cnt_ = 0;
  }
  index_t cnt_{0};
  buffer_type x_;
  Scalar difficulty_;
};
}  // namespace mathprim::optim::ex_probs