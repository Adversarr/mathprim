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
  explicit quad_problem(index_t dsize, index_t groups = 1) {
    // set A to randomly diagonal dominance
    for (index_t i = 0; i < groups; ++i) {
      A_.emplace_back(make_buffer<Scalar>(dsize, dsize));
      auto A = A_.back().view();
      for (auto [i, j] : A.shape()) {
        A(i, j) = i == j ? dsize : static_cast<Scalar>(i + j) / dsize;
      }
      parameters_.emplace_back(make_buffer<Scalar>(dsize));
      base::register_parameter(parameters_.back().view());
    }
  }

  quad_problem(quad_problem&&) noexcept = default;
  quad_problem& operator=(quad_problem&&) noexcept = default;

protected:
  void on_setup() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto x = eigen_support::cmap(parameters_[i].view());
      x.setRandom();
    }
  }

  void eval_value_and_gradients_impl() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto& A = A_[i];
      auto a = eigen_support::cmap(A.const_view());
      auto x = eigen_support::cmap(parameters_[i].view());
      auto ax = eigen_support::cmap(base::at(i).gradient());
      ax.noalias() = a * x;
      base::accumulate_loss(0.5 * x.dot(ax));
    }


    // auto a = eigen_support::cmap(A_.const_view());
    // auto x = eigen_support::cmap(parameters_.view());
    // auto ax = eigen_support::cmap(base::at(0).gradient());
    // ax.noalias() = a * x;
    // base::accumulate_loss(0.5 * x.dot(ax));
  }

  void eval_value_impl() { eval_value_and_gradients_impl(); }

  void eval_gradients_impl() { eval_value_and_gradients_impl(); }

  std::vector<buffer_type> parameters_;
  std::vector<contiguous_matrix_buffer<Scalar, device::cpu>> A_;
};

}  // namespace mathprim::optim::ex_probs