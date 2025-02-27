#pragma once
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/supports/eigen_dense.hpp"

namespace mathprim::optim::ex_probs {

// All optimizer should work on this simple problem
template <typename Scalar>
class quad_problem : public basic_problem<quad_problem<Scalar>, Scalar, device::cpu> {
public:
  using base = basic_problem<quad_problem<Scalar>, Scalar, device::cpu>;
  friend base;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using buffer_type = typename base::buffer_type;
  using parameter = typename base::parameter;

  // It formulates a quadratic problem with the following form:
  // f(x) = 0.5 * x^T * A * x
  explicit quad_problem(index_t dsize) : parameters_(make_buffer<Scalar>(dsize)), A_(make_buffer<Scalar>(dsize, dsize)) {
    // set A to randomly diagonal dominance
    auto dd = static_cast<Scalar>(dsize);
    for (auto [i, j] : A_.shape()) {
      auto ij = static_cast<Scalar>(i + j) / dd;
      A_.view()(i, j) = i == j ? dd : ij;
    }
    base::register_parameter(parameters_.view(), "x");
  }

  quad_problem(quad_problem&&) noexcept = default;
  quad_problem& operator=(quad_problem&&) noexcept = default;

protected:
  void on_setup() {
    auto x = eigen_support::cmap(parameters_.view());
    x.setRandom();
  }

  void eval_value_and_gradients_impl() {
    auto a = eigen_support::cmap(A_.const_view());
    auto x = eigen_support::cmap(parameters_.view());
    auto ax = eigen_support::cmap(base::at(0).gradient());
    ax.noalias() = a * x;
    base::accumulate_loss(0.5 * x.dot(ax));
  }

  void eval_value_impl() { eval_value_and_gradients_impl(); }

  void eval_gradients_impl() { eval_value_and_gradients_impl(); }


  buffer_type parameters_;
  contiguous_matrix_buffer<Scalar, device::cpu> A_;
};

}  // namespace mathprim::optim::ex_probs