/**
 * @brief L-BFGS Optimizer.
 * @ref   https://en.wikipedia.org/wiki/Limited-memory_BFGS
 *
 * >>> Two-loop recursion for L-BFGS >>>
 * |    q <- grad_k
 * |    for i = k-1, k-2, ..., k-m
 * |      rho_i <- 1 / (s_i^T y_i)
 * |      alpha_i <- rho_i s_i^T q
 * |      q <- q - alpha_i y_i
 * |    z <- H_0 q ==> Preconditioner of L-BFGS
 * |    for i = k-m, k-m+1, ..., k-1
 * |      beta <- rho_i y_i^T z
 * |      z <- z + s_i (alpha_i - beta)
 * |    return -z
 * <<< Two-loop recursion for L-BFGS <<<
 *
 *     for the default preconditioner, it use `gamma I`, where
 *       - gamma = y^T s / y^T y,
 *       - s y is the last avaialble history
 */
#pragma once
#include "mathprim/blas/blas.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/linesearcher/backtracking.hpp"
#include <algorithm>
#include <cmath>

namespace mathprim::optim {

template <typename Derived, typename Scalar, typename Device>
struct l_bfgs_preconditioner {
  using vector_type = contiguous_vector_view<Scalar, Device>;
  using const_vector = contiguous_vector_view<const Scalar, Device>;
  l_bfgs_preconditioner() = default;
  MATHPRIM_INTERNAL_MOVE(l_bfgs_preconditioner, default);

  void apply(vector_type z, const_vector q, const_vector s, const_vector y) {
    static_cast<Derived*>(this)->apply_impl(z, q, s, y);
  }
};

template <typename Scalar, typename Device, typename Blas>
struct l_bfgs_preconditioner_identity
    : l_bfgs_preconditioner<l_bfgs_preconditioner_identity<Scalar, Device, Blas>, Scalar, Device> {
  using base = l_bfgs_preconditioner<l_bfgs_preconditioner_identity<Scalar, Device, Blas>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  l_bfgs_preconditioner_identity() = default;
  MATHPRIM_INTERNAL_MOVE(l_bfgs_preconditioner_identity, default);

  void apply_impl(vector_type z, const_vector q, const_vector /* s */, const_vector /* y */) {
    blas_.copy(z, q);  // z <- q
  }

  Blas blas_;
};

template <typename Scalar, typename Device, typename Blas>
struct l_bfgs_preconditioner_default
    : l_bfgs_preconditioner<l_bfgs_preconditioner_default<Scalar, Device, Blas>, Scalar, Device> {
  using base = l_bfgs_preconditioner<l_bfgs_preconditioner_default<Scalar, Device, Blas>, Scalar, Device>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;

  l_bfgs_preconditioner_default() = default;
  MATHPRIM_INTERNAL_MOVE(l_bfgs_preconditioner_default, default);

  void apply_impl(vector_type z, const_vector q, const_vector s, const_vector y) {
    blas::basic_blas<Blas, Scalar, Device>& bl = blas_;
    if (!s || !y) {
      // no history, no preconditioning.
      bl.copy(z, q);
      return;
    } else {
      Scalar gamma = bl.dot(y, s) / bl.dot(y, y);
      bl.copy(z, q);      // z <- q
      bl.scal(gamma, z);  // z <- gamma * q
    }
  }

  Blas blas_;
};

/**
 * @brief L-BFGS Optimizer.
 * @ref   https://en.wikipedia.org/wiki/Limited-memory_BFGS
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Blas
 */
template <typename Scalar, typename Device, typename Blas,
          typename Linesearcher = backtracking_linesearcher<Scalar, Device, Blas>,
          typename Preconditioner = l_bfgs_preconditioner_default<Scalar, Device, Blas>>
class l_bfgs_optimizer
    : public basic_optimizer<l_bfgs_optimizer<Scalar, Device, Blas, Linesearcher, Preconditioner>, Scalar, Device> {
public:
  using base = basic_optimizer<l_bfgs_optimizer<Scalar, Device, Blas, Linesearcher, Preconditioner>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;

  l_bfgs_optimizer() = default;
  MATHPRIM_INTERNAL_MOVE(l_bfgs_optimizer, default);

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback) {
    blas::basic_blas<Blas, Scalar, Device>& bl = blas_;
    l_bfgs_preconditioner<Preconditioner, Scalar, Device>& prec = preconditioner_;
    basic_linesearcher<Linesearcher, Scalar, Device>& ls = linesearcher_;

    auto criteria = base::criteria();
    auto grads = problem.fused_gradients();
    result_type result;
    Scalar& value = result.value_;
    Scalar& last_change = result.last_change_;
    Scalar& grad_norm = result.grad_norm_;
    index_t& iteration = result.iterations_;
    bool &converged = result.converged_;


    // 1. prepare all the buffers.
    setup_buffers(grads.numel());
    value = problem.eval_value_and_gradients();
    grad_norm = bl.norm(grads);
    converged = grad_norm < criteria.tol_grad_;
    if (converged) {
      // lucky path.
      return result;
    }

    // 3. main loop.
    auto q = q_.view();
    auto z = z_.view();
    auto sn = s_new_.view();
    auto yn = y_new_.view();
    for (; iteration < criteria.max_iterations_; ++ iteration) {
      callback(result);
      // q <- grad_k
      bl.copy(q, grads);
      bl.copy(yn, grads); // y <- grad old

      two_loop_step_in(); // q loop

      // Apply the preconditioner.
      if (memory_avail_ == 0) {
        prec.apply(z, q, {}, {});
      } else {
        index_t rotated = (memory_start_ + memory_avail_ - 1) % memory_size_;
        auto s_latest = s_.const_view()[rotated], y_latest = y_.const_view()[rotated];
        prec.apply(z, q, s_latest, y_latest);
      }

      two_loop_step_out(); // z loop

      // Launch linesearcher.
      auto [ls_result, ls_step_size] = ls.search(problem, z, learning_rate_);

      // s_new <- x_k+1 - x_k = -alpha z.
      bl.copy(sn, z);
      bl.scal(-ls_step_size, sn);
      // update parameters.
      problem.for_each_parameter([this, &sn, &bl, alpha = ls_step_size](auto& item) {
        auto& value = item.value();
        const index_t offset = item.offset();
        const index_t size = value.size();
        auto delta = sn.sub(offset, offset + size);
        bl.axpy(1, delta, value);
      });

      Scalar new_value = problem.eval_value_and_gradients();
      last_change = value - new_value;
      value = new_value;
      grad_norm = bl.norm(grads);
      if (last_change < criteria.tol_change_ && last_change >= 0) {
        converged = true;
        break;
      }

      bl.scal(-1, yn);        // y_new <- -grad_k
      bl.axpy(1, grads, yn);  // y_new <- grad_k+1 - grad_k
      push_memory();
    }

    // 4. check & return
    if (!converged) {
      fprintf(stderr, "Warning: L-BFGS optimizer did not converge.\n");
    }
    return result;
  }

  void two_loop_step_in(){
    // operates on q
    auto q = q_.view();
    auto s = s_.const_view(), y = y_.const_view();
    for (index_t i = memory_avail_ - 1; i >= 0; --i) {
      // For i = k-1, k-2, ..., k-m
      const index_t rotated = (memory_start_ + i) % memory_size_;
      const Scalar rho = rho_[static_cast<size_t>(rotated)];
      const Scalar alpha = alpha_[static_cast<size_t>(rotated)] = (rho * blas_.dot(s[rotated], q));
      blas_.axpy(-alpha, y[rotated], q);
    }
  }

  void two_loop_step_out() {
    // operates on z
    auto z = z_.view();
    auto s = s_.const_view(), y = y_.const_view();
    for (index_t i = 0; i < memory_avail_; ++i) {
      // For i = k-m, k-m+1, ..., k-1
      const index_t rotated = (memory_start_ + i) % memory_size_;
      const Scalar rho = rho_[static_cast<size_t>(rotated)];
      const Scalar beta = beta_[static_cast<size_t>(rotated)] = rho * blas_.dot(y[rotated], z);
      const Scalar alpha_minus_beta = alpha_[static_cast<size_t>(rotated)] - beta;
      blas_.axpy(alpha_minus_beta, s[rotated], z);
    }
  }

  void push_memory() {
    index_t targ;
    if (memory_avail_ < memory_size_) {
      MATHPRIM_ASSERT(memory_start_ == 0 && "Internal logic error.");
      targ = memory_avail_;
      ++memory_avail_;
    } else {
      // discard the oldest.
      targ = memory_start_;
      memory_start_ = (memory_start_ + 1) % memory_size_;
    }
    auto sn = s_new_.const_view(), yn = y_new_.const_view();
    rho_[targ] = 1 / blas_.dot(sn, yn);
    blas_.copy(s_.view()[targ], sn);
    blas_.copy(y_.view()[targ], yn);
  }

  void setup_buffers(index_t ndofs) {
    s_ = make_buffer<Scalar, Device>(memory_size_, ndofs);
    y_ = make_buffer<Scalar, Device>(memory_size_, ndofs);
    s_new_ = make_buffer<Scalar, Device>(ndofs);
    y_new_ = make_buffer<Scalar, Device>(ndofs);
    q_ = make_buffer<Scalar, Device>(ndofs);
    z_ = make_buffer<Scalar, Device>(ndofs);
    s_.fill_bytes(0);
    y_.fill_bytes(0);
    rho_.resize(memory_size_, 0);
    alpha_.resize(memory_size_, 0);
    beta_.resize(memory_size_, 0);
  }

  //// history buffers ////
  contiguous_matrix_buffer<Scalar, Device> s_, y_;                  // rotating buffer for histories
  std::vector<Scalar> rho_, alpha_, beta_;                          // histories
  index_t memory_start_{0};                                         // latest history index
  index_t memory_avail_{0};                                         // available history count
  //// working buffers ////
  contiguous_vector_buffer<Scalar, Device> q_, z_, s_new_, y_new_;
  Blas blas_;

public:  // Hyper parameters.
  Preconditioner preconditioner_;  ///< Preconditioner for L-BFGS, default is scaled identity.
  Linesearcher linesearcher_;      ///< Linesearcher for L-BFGS, for better convergency, consider wolfe.
  Scalar learning_rate_{1.0};      ///< learning rate of L-BFGS, due to linesearch, 1.0 is a good start.
  index_t memory_size_{10};        ///< Number of previous steps to store in memory, i.e. m in L-BFGS.
};
}  // namespace mathprim::optim