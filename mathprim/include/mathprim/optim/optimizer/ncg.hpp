#pragma once
#include <cmath>
#include <stdexcept>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/linesearcher/backtracking.hpp"

namespace mathprim::optim {

template <typename Derived, typename Scalar, typename Device>
struct ncg_preconditioner {
  using vector_type = contiguous_vector_view<Scalar, Device>;
  using const_vector = contiguous_vector_view<const Scalar, Device>;
  ncg_preconditioner() = default;
  MATHPRIM_INTERNAL_MOVE(ncg_preconditioner, default);

  void apply(vector_type x, const_vector g) { static_cast<Derived*>(this)->apply_impl(x, g); }
};

template <typename Scalar, typename Device>
struct ncg_preconditioner_identity
    : public ncg_preconditioner<ncg_preconditioner_identity<Scalar, Device>, Scalar, Device> {
  using base = ncg_preconditioner<ncg_preconditioner_identity<Scalar, Device>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  ncg_preconditioner_identity() = default;
  MATHPRIM_INTERNAL_MOVE(ncg_preconditioner_identity, default);

  void apply_impl(vector_type x, const_vector g) {
    copy(x, g);  // x <- g
  }
};

enum class ncg_strategy {
  fletcher_reeves,
  polak_ribiere,
  hestenes_stiefel,
  dai_yuan,
  polak_ribiere_clamped,
  hestenes_stiefel_clamped,
};

/**
 * @brief Nonlinear Conjugate Gradient optimizer
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Blas
 */
template <typename Scalar, typename Device, typename Blas,
          typename Linesearcher = backtracking_linesearcher<Scalar, Device, Blas>,
          typename Preconditioner = ncg_preconditioner_identity<Scalar, Device>>
class ncg_optimizer
    : public basic_optimizer<ncg_optimizer<Scalar, Device, Blas, Linesearcher, Preconditioner>, Scalar, Device> {
public:
  using base = basic_optimizer<ncg_optimizer<Scalar, Device, Blas, Linesearcher, Preconditioner>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;
  using temp_buffer = contiguous_vector_buffer<Scalar, Device>;

  ncg_optimizer() = default;
  MATHPRIM_INTERNAL_MOVE(ncg_optimizer, default);

  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback) {
    blas::basic_blas<Blas, Scalar, Device>& bl = blas_;
    ncg_preconditioner<Preconditioner, Scalar, Device>& prec = preconditioner_;
    basic_linesearcher<Linesearcher, Scalar, Device>& ls = linesearcher_;

    auto criteria = base::criteria();
    auto grads = problem.fused_gradients();
    result_type result;
    Scalar& value = result.value_;
    Scalar& last_change = result.last_change_;
    Scalar& grad_norm = result.grad_norm_;
    index_t& iteration = result.iterations_;
    bool& converged = result.converged_;
    // 1. prepare all the buffers.
    s_ = make_buffer<Scalar, Device>(grads.numel());
    d_ = make_buffer<Scalar, Device>(grads.numel());
    auto r = problem.fused_gradients();
    auto s = s_.view(), d = d_.view();
    value = problem.eval_value_and_gradients();
    grad_norm = bl.norm(grads);
    converged = grad_norm < criteria.tol_grad_;
    if (converged) {
      // lucky path.
      return result;
    }

    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(value), std::runtime_error,
                                  "Initial value is not finite. Value: " << value);
    MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(grad_norm), std::runtime_error,  //
                                  "Initial gradient norm is not finite.");

    // 2. init
    prec.apply(s, r);  // d <- M^-1 * r
    copy(d, s);
    Scalar delta_new = bl.dot(r, d);
    Scalar beta = 0;
    index_t restart_counter = 0;

    // Validate restart_period_ to avoid division by zero.
    MATHPRIM_INTERNAL_CHECK_THROW(restart_period_ > 0, std::runtime_error, "restart_period_ must be greater than zero.");
    MATHPRIM_INTERNAL_CHECK_THROW(delta_new > 0, std::runtime_error,
                                  "Initial search direction is not a descent direction (delta_new >= 0).");

    // 3. main loop.
    for (; iteration < criteria.max_iterations_; ++iteration) {
      callback(result);
      const Scalar expected_descent = bl.dot(r, d);  // grad[n-1] dot d[n-1]
      try {
        ls.search(problem, d, learning_rate_);
      } catch (const std::runtime_error& e) {
        // Reinit d = r
        copy(d, r);
        ls.restore_state(problem, true);
        ls.search(problem, d, learning_rate_);
      }
      auto new_value = problem.eval_value_and_gradients();
      last_change = value - new_value;
      value = new_value;
      grad_norm = bl.norm(grads);

      converged = grad_norm < criteria.tol_grad_;
      converged |= (last_change < criteria.tol_change_ && last_change >= 0);
      MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(value), std::runtime_error,
                                    "At step " << iteration << ", value is not finite.");
      MATHPRIM_INTERNAL_CHECK_THROW(std::isfinite(grad_norm), std::runtime_error,
                                    "At step " << iteration << ", gradient norm is not finite.");
      if (converged) {
        break;
      }
      const Scalar eps = std::numeric_limits<Scalar>::epsilon();
      const Scalar delta_old = delta_new + eps;  // grad[n-1] dot s[k-1]
      const Scalar delta_mid = bl.dot(r, s);     // grad[n] dot s[n-1]
      prec.apply(s, r);                          // s = M^-1 * r
      delta_new = bl.dot(r, s);                  // grad[n] dot s[n]
      switch (strategy_) {
        case ncg_strategy::fletcher_reeves: {
          beta = delta_new / delta_old;
          break;
        }
        case ncg_strategy::polak_ribiere: {
          beta = (delta_new - delta_mid) / delta_old;
          break;
        }
        case ncg_strategy::polak_ribiere_clamped: {
          beta = (delta_new - delta_mid) / delta_old;
          beta = std::max<Scalar>(beta, 0);
          break;
        }

        // ed_mid = grad[n] dot search_dir[n-1]
        case ncg_strategy::hestenes_stiefel: {
          Scalar ed_mid = bl.dot(r, d);
          beta = (delta_new - delta_mid) / (ed_mid - expected_descent);
          break;
        }
        case ncg_strategy::hestenes_stiefel_clamped: {
          Scalar ed_mid = bl.dot(r, d);
          beta = (delta_new - delta_mid) / (ed_mid - expected_descent);
          beta = std::max<Scalar>(beta, 0);
          break;
        }
        case ncg_strategy::dai_yuan: {
          Scalar ed_mid = bl.dot(r, d);
          beta = (delta_new) / (ed_mid - expected_descent);
          break;
        }
        default:
          MATHPRIM_UNREACHABLE();
      }

      MATHPRIM_INTERNAL_CHECK_THROW(!std::isnan(beta), std::runtime_error, "beta computation resulted in NaN");

      // d <- s + beta d
      bl.axpby(1, s, beta, d);
      if ((restart_counter + 1) % restart_period_ == 0) {
        copy(d, s);
        restart_counter = 0;
      } else {
        ++restart_counter;
      }
    }

    // 4. check & return
    if (!converged) {
      fprintf(stderr, "Warning: Nonlinear Conjugate Gradient optimizer did not converge.\n");
    }
    return result;
  }

  // Hyper parameters.
  ncg_strategy strategy_{ncg_strategy::fletcher_reeves};
  Blas blas_;
  Preconditioner preconditioner_;  ///< Preconditioner, default is scaled identity.
  Linesearcher linesearcher_;      ///< Linesearcher, for better convergency, consider wolfe.
  Scalar learning_rate_{1.0};      ///< Learning rate, due to linesearch, 1.0 is a good start.
  temp_buffer s_, d_;
  index_t restart_period_{1000};
};

}  // namespace mathprim::optim
