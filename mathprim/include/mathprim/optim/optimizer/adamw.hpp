#pragma once
#include <algorithm>
#include <cmath>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/optim/basic_optim.hpp"

#ifdef __CUDACC__
#  include "mathprim/parallel/cuda.cuh"
#endif

namespace mathprim::optim {

namespace internal {
template <typename Scalar, typename Device>
struct momentum_modification {
  static_assert(mathprim::internal::always_false_v<Scalar>, "Not implemented.");
};

template <typename Scalar>
struct momentum_modification<Scalar, device::cpu> {
  static void inplace_sqr(const contiguous_vector_view<Scalar, device::cpu>& v) {
    auto total = v.size();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      v[i] = v[i] * v[i];
    }
  }
  
  static void do_direction(const contiguous_vector_view<Scalar, device::cpu>& m,
                    const contiguous_vector_view<Scalar, device::cpu>& v,
                    const contiguous_vector_view<Scalar, device::cpu>& out,
                    Scalar eps) {
    auto total = m.size();
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (index_t i = 0; i < total; ++i) {
      out[i] = m[i] / (::sqrt(v[i]) + eps);
    }
  }
};

#ifdef __CUDACC__
template <typename Scalar>
struct momentum_modification<Scalar, device::cuda> {
  static void inplace_sqr(const contiguous_vector_view<Scalar, mathprim::device::cuda>& v) {
    par::cuda().run((v.shape()), [v]__device__(index_t i) {
      v[i] = v[i] * v[i];
    });
  }

  static void do_direction(const contiguous_vector_view<Scalar, mathprim::device::cuda>& m,
                    const contiguous_vector_view<Scalar, mathprim::device::cuda>& v,
                    const contiguous_vector_view<Scalar, mathprim::device::cuda>& out,
                    Scalar eps) {
    par::cuda().run((m.shape()), [m, v, out, eps]__device__(index_t i) {
      out[i] = m[i] / (::sqrt(v[i]) + eps);
    });
  }
};
#endif
}  // namespace internal

/**
 * @brief AdamW optimizer.
 * @ref   https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#adamw
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Problem
 * @tparam Blas
 */
template <typename Scalar, typename Device, typename Blas>
class adamw_optimizer : public basic_optimizer<adamw_optimizer<Scalar, Device, Blas>, Scalar, Device> {
public:
  using base = basic_optimizer<adamw_optimizer<Scalar, Device, Blas>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;

  adamw_optimizer() = default;

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback) {
    blas::basic_blas<Blas, Scalar, Device>& bl = blas_;
    auto criteria = base::criteria();
    auto gradients_view = problem.fused_gradients();
    result_type result;
    Scalar& value = result.value_;
    Scalar& last_change = result.last_change_;
    Scalar& grad_norm = result.grad_norm_;
    index_t& iteration = result.iterations_;
    bool& converged = result.converged_;
    // Initialize the momentum buffer.
    first_mom_ = make_buffer<Scalar, Device>(gradients_view.shape());
    second_mom_ = make_buffer<Scalar, Device>(gradients_view.shape());
    first_mom_corr_ = make_buffer<Scalar, Device>(gradients_view.shape());
    second_mom_corr_ = make_buffer<Scalar, Device>(gradients_view.shape());
    first_mom_.fill_bytes(0);
    second_mom_.fill_bytes(0);

    value = problem.eval_value_and_gradients();
    last_change = std::numeric_limits<Scalar>::infinity();
    grad_norm = bl.norm(gradients_view);
    iteration = 0;
    converged = grad_norm < criteria.tol_grad_;
    if (converged) {
      return result;
    }

    // If value/|grad| is nan, fast return.
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(value), std::runtime_error, "Initial value is not finite.");
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(grad_norm), std::runtime_error, "Initial gradient norm is not finite.");

    auto m = first_mom_.view(), v = second_mom_.view();
    auto m_hat = first_mom_corr_.view(), v_hat = second_mom_corr_.view();
    for (; (iteration < criteria.max_iterations_) && (grad_norm >= criteria.tol_grad_); ++iteration) {
      callback(result);
      // theta_t <- theta_{t-1} - gamma lambda theta_{t-1}
      if (weight_decay_ > 0) {
        problem.for_each_parameter([&bl, lr = learning_rate_, wd = weight_decay_](auto& param) {
          auto& value = param.value();
          bl.scal(1 - wd * lr, value);
        });
      }

      // m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
      bl.axpby(1 - beta1_, gradients_view, beta1_, m);

      // v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t^2
      // first dot inplace square gradients to get g_t^2
      internal::momentum_modification<Scalar, Device>::inplace_sqr(gradients_view);
      bl.axpby(1 - beta2_, gradients_view, beta2_, v);

      // ?: Use double precision in this part to avoid numerical instability.
      auto bias_correction1 = static_cast<Scalar>(1 - std::pow(static_cast<Scalar>(beta1_), iteration + 1));
      auto bias_correction2 = static_cast<Scalar>(1 - std::pow(static_cast<Scalar>(beta2_), iteration + 1));
      bl.copy(m_hat, m);
      bl.copy(v_hat, v);
      bl.scal(1 / bias_correction1, m_hat);  // m_hat <- m_t / (1 - beta1^t)
      bl.scal(1 / bias_correction2, v_hat);  // v_hat <- v_t / (1 - beta2^t)

      internal::momentum_modification<Scalar, Device>::do_direction(m_hat, v_hat, gradients_view, epsilon_);

      // Update the parameters: since we store the m_hat / (sqrt(v) + eps) in gradients_view, we can directly update
      // the parameters using the gradients.
      problem.for_each_parameter([&bl, lr = learning_rate_](auto& param) {
        auto& value = param.value();
        auto& grad = param.gradient();
        bl.axpy(-lr, grad, value);
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

  contiguous_vector_buffer<Scalar, Device> first_mom_, second_mom_;
  contiguous_vector_buffer<Scalar, Device> first_mom_corr_, second_mom_corr_;
  Blas blas_;

public:  // Hyper parameters.
  Scalar learning_rate_{1e-2};
  Scalar beta1_{0.9};
  Scalar beta2_{0.95};
  Scalar epsilon_{1e-8};
  // bool amsgrad_{false};
  Scalar weight_decay_{0};
};

}  // namespace mathprim::optim