#pragma once
#include <cmath>

#include "mathprim/blas/blas.hpp"
#include "mathprim/optim/basic_optim.hpp"

namespace mathprim::optim {

/**
 * @brief Gradient Descent Optimizer, with optional momentum.
 * @ref   https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Problem
 * @tparam Blas
 */
template <typename Scalar, typename Device, typename Blas,
          typename Linesearch = no_linesearcher<Scalar, Device>>  // Implementation of the optimizer
class gradient_descent_optimizer
    : public basic_optimizer<gradient_descent_optimizer<Scalar, Device, Blas, Linesearch>, Scalar, Device> {
public:
  using base = basic_optimizer<gradient_descent_optimizer<Scalar, Device, Blas, Linesearch>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;
  static constexpr bool no_linesearcher_v = std::is_same_v<no_linesearcher<Scalar, Device>, Linesearch>;

  gradient_descent_optimizer() = default;

  Linesearch& linesearcher() noexcept { return linesearcher_; }

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback && callback) {
    blas::basic_blas<Blas, Scalar, Device>& bl = blas_;
    basic_linesearcher<Linesearch, Scalar, Device>& ls = linesearcher_;
    auto criteria = base::criteria();
    auto gradients_view = problem.fused_gradients();
    // First, we consider the most simple case, without momentum.
    result_type result;
    Scalar& value = result.value_;
    Scalar& last_change = result.last_change_;
    Scalar& grad_norm = result.grad_norm_;
    index_t& iteration = result.iterations_;
    bool& converged = result.converged_;

    Scalar momentum_value = momentum_;
    if constexpr (!no_linesearcher_v) {
      // If we have a linesearcher, we need to disable momentum.
      if (momentum_value != 0) {
        fprintf(stderr, "Warning: Momentum is disabled when using linesearcher.\n");
        momentum_value = 0;
      }
    } else {
      // Initialize the momentum buffer.
      momentum_buffer_ = make_buffer<Scalar, Device>(gradients_view.shape());
      momentum_buffer_.fill_bytes(0);
    }

    value = problem.eval_value_and_gradients();
    last_change = std::numeric_limits<Scalar>::infinity();
    grad_norm = bl.norm(gradients_view);
    iteration = 0;
    converged = grad_norm < criteria.tol_grad_;
    if (converged) {
      // lucky path.
      return result;
    }

    // If value/|grad| is nan, fast return.
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(value), std::runtime_error, "Initial value is not finite.");
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(grad_norm), std::runtime_error, "Initial gradient norm is not finite.");

    auto mom_view = momentum_buffer_.view();
    for (; (iteration < criteria.max_iterations_) && (grad_norm >= criteria.tol_grad_);
         ++iteration) {
      callback(result);
      Scalar alpha;
      if constexpr (no_linesearcher_v) {
        // Momentum.
        if (momentum_value != 0) {
          if (iteration == 0) {
            // b <- g
            copy(mom_view, gradients_view);
          } else {
            // b <- (1-damping) g + momentum b
            bl.scal(momentum_, mom_view);                    // momentum b
            bl.axpy(1 - damping_, gradients_view, mom_view); // (1-damping) g + momentum b
          }

          if (nesterov_) {
            // g <- g + u b_t
            bl.axpy(momentum_, mom_view, gradients_view);
          } else {
            // g <- b_t
            copy(gradients_view, mom_view);
          }
        }

        alpha = learning_rate_;
      } else {
        auto [ls_result, ls_step_size] = ls.template search<ProblemDerived>(problem, gradients_view, learning_rate_);
        alpha = ls_step_size;
      }

      problem.for_each_parameter([&bl, alpha] (auto& param) {
        auto& value = param.value();
        auto& gradient = param.gradient();
        bl.axpy(-alpha, gradient, value);
      });

      Scalar new_value = problem.eval_value_and_gradients();  // Update the gradients.
      last_change = value - new_value;                        // minimize => change > 0
      value = new_value;
      grad_norm = bl.norm(gradients_view);
      if (last_change < criteria.tol_change_ && last_change >= 0) {
        break;
      }
    }
    return result;
  }

  contiguous_vector_buffer<Scalar, Device> momentum_buffer_;
  Linesearch linesearcher_;
  Blas blas_;

public: // Hyper parameters.
  Scalar learning_rate_{1e-2};
  Scalar momentum_{0};     // momentum rate: 0 to disable
  Scalar damping_{0};      // damping of momentum
  bool nesterov_{false};    // whether to use nystrom momentum
  // Scalar weight_decay_;
};

}  // namespace mathprim::optim