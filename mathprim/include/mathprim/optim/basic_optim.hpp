#pragma once
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "mathprim/blas/blas.hpp"
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/core/utils/parameter_item.hpp"
#include "mathprim/core/view.hpp"

namespace mathprim::optim {

namespace internal {

template <typename Real>
Real solve_optimal_step_size_quadratic(Real f_lo, Real g_lo, Real f_hi, Real lo, Real hi) {
  // See Page 58, Numerical Optimization, Nocedal and Wright. Eq. 3.57
  // f(x) = B (x-a)^2 + C (x-a) + D
  // f(a) = D.
  // f'(a) = C.
  // f(b) = B (b-a)^2 + C (b-a) + D => B = (f(b) - f(a) - C (b-a)) / (b-a)^2
  // optimal = a - C / (2 B)

  const Real b = hi, a = lo, df_a = g_lo, f_a = f_lo, f_b = f_hi;
  Real b_a = b - a;
  Real c = df_a /* , d = f_a */;
  Real bb = (f_b - f_a - c * b_a) / (b_a * b_a);
  return a - c / (2 * bb);
}

template <typename Real>
Real solve_optimal_step_size_none(Real shrink, Real lo, Real hi) {
  return shrink * hi + (1 - shrink) * lo;
}

}  // namespace internal

/**
 * @brief Base class for all optimization problems.
 *
 * @tparam Derived
 * @tparam Scalar
 * @tparam Device
 */
template <typename Derived,  // The actual problem class
          typename Scalar, typename Device>
class basic_problem {
  // Note: the child class have to implements these problems.
  // void eval_gradients_impl() {}
  // void eval_value_impl() {}
public:
  using view_type = contiguous_view<Scalar, dshape<1>, Device>;
  using const_view = contiguous_view<const Scalar, dshape<1>, Device>;
  using buffer_type = contiguous_buffer<Scalar, dshape<1>, Device>;
  using parameter = parameter_item<Scalar, Device>;
  using parameter_container = std::vector<parameter>;

  // Generally, you cannot have a copy constructor for a optimization problem.
  basic_problem() = default;
  MATHPRIM_INTERNAL_MOVE(basic_problem, default);
  MATHPRIM_INTERNAL_COPY(basic_problem, delete);

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  Scalar eval_value() {
    loss_ = 0;
    derived().eval_value_impl();
    return loss_;
  }

  void eval_gradients() {
    zero_gradients();
    derived().eval_gradients_impl();
  }

  // Fused evaluation of value and gradients
  Scalar eval_value_and_gradients() {
    loss_ = 0;
    zero_gradients();
    derived().eval_value_and_gradients_impl();
    return loss_;
  }

  Scalar current_value() const noexcept { return loss_; }

  void setup() {
    prepare_fused_gradients();
    derived().on_setup();
  }

  Scalar loss() const noexcept { return loss_; }

  view_type fused_gradients() noexcept { return fused_gradients_.view(); }

  const_view fused_gradients() const noexcept { return fused_gradients_.const_view(); }

  parameter_container& parameters() noexcept { return parameters_; }
  parameter& at(index_t index) noexcept { return parameters_.at(index); }

  template <typename Fn>
  void for_each_parameter(Fn&& fn) {
    std::for_each(parameters_.begin(), parameters_.end(), std::forward<Fn>(fn));
  }

protected:
  /// @brief be called after the parent class setup(optional for child class)
  void on_setup() {}

  /// @brief (during optimization) accumulate the loss
  void accumulate_loss(Scalar item) noexcept { loss_ += item; }

  /// @brief Reset the gradients of the parameters
  void zero_gradients() { fused_gradients_.fill_bytes(0); }

  /// @brief creates the fused gradients buffer, and set the gradients of the parameters
  void prepare_fused_gradients() noexcept {
    index_t total_size = 0;
    for (const auto& param : parameters_) {
      total_size += param.value().size();
    }
    fused_gradients_ = make_buffer<Scalar, Device>(total_size);
    zero_gradients();
    for (index_t i = 0, offset = 0; static_cast<size_t>(i) < parameters_.size(); ++i) {
      auto& param = parameters_[i];
      index_t size = param.value().size();
      param.set_gradient(fused_gradients_.view().sub(offset, offset + size));
      param.set_offset(offset);
      offset += size;
    }
  }

  /// @brief Register a parameter to the problem
  index_t register_parameter(view_type value, std::string name = "") noexcept {
    parameters_.emplace_back(value, name);
    return static_cast<index_t>(parameters_.size() - 1);
  }

  /// @brief Register a parameter to the problem
  void eval_value_and_gradients_impl() {
    derived().eval_value_impl();
    derived().eval_gradients_impl();
  }

private:
  buffer_type fused_gradients_;  // fused gradients
  Scalar loss_{};
  parameter_container parameters_;
};

template <typename Scalar>
struct optim_result {
  Scalar value_{std::numeric_limits<Scalar>::quiet_NaN()};
  Scalar last_change_{std::numeric_limits<Scalar>::quiet_NaN()};
  Scalar grad_norm_{std::numeric_limits<Scalar>::quiet_NaN()};
  int iterations_{0};
  bool converged_{false};

  optim_result() = default;
  MATHPRIM_INTERNAL_COPY(optim_result, default);
  MATHPRIM_INTERNAL_MOVE(optim_result, default);
};

template <typename Scalar>
struct stopping_criteria {
  Scalar tol_change_{0};     // |f[x] - f[x_prev]| < tol_change => stop
  Scalar tol_grad_{1e-3};    // |g| < tol_grad => stop
  int max_iterations_{100};  // maximum number of iterations
};

template <typename Derived, typename Scalar, typename Device>  // Implementation of the optimizer
class basic_optimizer {
public:
  using view_type = contiguous_view<Scalar, dshape<1>, Device>;
  using const_view = contiguous_view<const Scalar, dshape<1>, Device>;
  using stopping_criteria_type = stopping_criteria<Scalar>;
  using result_type = optim_result<Scalar>;

  basic_optimizer() = default;
  MATHPRIM_INTERNAL_MOVE(basic_optimizer, default);
  MATHPRIM_INTERNAL_COPY(basic_optimizer, delete);

  struct do_nothing_cb {
    inline void operator()(const result_type&) {}
  };

  template <typename ProblemDerived, typename Callback = do_nothing_cb>
  result_type optimize(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback = {}) {
    return static_cast<Derived&>(*this).template optimize_impl<ProblemDerived, Callback>(
        problem, std::forward<Callback>(callback));
  }

  stopping_criteria_type& criteria() noexcept { return stopping_criteria_; }

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }

  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  stopping_criteria_type stopping_criteria_;
};

template <typename Derived, typename Scalar, typename Device>
class basic_linesearcher : public basic_optimizer<basic_linesearcher<Derived, Scalar, Device>, Scalar, Device> {
public:
  using base = basic_optimizer<basic_linesearcher<Derived, Scalar, Device>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using result_type = typename base::result_type;
  basic_linesearcher() = default;
  MATHPRIM_INTERNAL_MOVE(basic_linesearcher, default);
  MATHPRIM_INTERNAL_COPY(basic_linesearcher, delete);

  view_type backuped() noexcept { return backuped_parameters_.view(); }
  const_view backuped() const noexcept { return backuped_parameters_.const_view(); }

  /**
   * @brief Search a step size along the search direction.
   *
   * Note: most algorithms does not provide the search direction, e.g.
   *   - gradient descent    => gradient
   *   - newton/quasi-newton => H^-1 grad
   * Therefore, we enforce the user to provide the search direction as the negative direction.
   * Note: linesearcher should exit at the optimal point.
   *
   * @tparam ProblemDerived
   * @tparam base::do_nothing_cb
   * @param problem
   * @param neg_search_dir
   * @param init_step
   * @param callback
   * @return std::pair<result_type, Scalar>
   */
  template <typename ProblemDerived, typename LinesearchCallback = typename base::do_nothing_cb>
  std::pair<result_type, Scalar> search(basic_problem<ProblemDerived, Scalar, Device>& problem,
                                        const_view neg_search_dir, Scalar init_step,
                                        LinesearchCallback&& callback = {}) {
    backup_state(problem);
    const Scalar cur_value = problem.current_value();
    neg_search_dir_ = neg_search_dir;
    step_size_ = init_step;
    auto result = base::template optimize<ProblemDerived, LinesearchCallback>(
        problem, std::forward<LinesearchCallback>(callback));
    const Scalar next_value = problem.current_value();
    // MATHPRIM_ASSERT(next_value <= cur_value && "Linesearcher should decrease the value.");
    MATHPRIM_INTERNAL_CHECK_THROW(next_value <= cur_value, std::runtime_error,
                                  "Linesearcher should decrease the value.");
    MATHPRIM_UNUSED(cur_value);
    MATHPRIM_UNUSED(next_value);

    // Note: linesearcher should exit at the optimal point.
    // restore_state(problem, true);
    return {result, step_size_};
  }

  /// @brief Restore the state of the problem.
  template <typename ProblemDerived>
  void restore_state(basic_problem<ProblemDerived, Scalar, Device>& problem, bool with_grad = false) {
    problem.for_each_parameter([this](auto& param) {
      auto& value = param.value();
      auto offset = param.offset();
      auto size = value.size();
      copy(value, backuped_parameters_.view().sub(offset, offset + size));
    });
    if (with_grad) {
      copy(problem.fused_gradients(), backuped_gradients_);
    }
  }

protected:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback) {
    return static_cast<Derived&>(*this).template optimize_impl<ProblemDerived, Callback>(
        problem, std::forward<Callback>(callback));
  }

  ///////////// These methods are for optimizer developers. //////////////

  /// @brief Backup the current state of the problem.
  template <typename ProblemDerived>
  void backup_state(basic_problem<ProblemDerived, Scalar, Device>& problem) {
    index_t total_params = problem.fused_gradients().numel();
    if (!backuped_parameters_ || backuped_parameters_.numel() != total_params) {
      backuped_parameters_ = make_buffer<Scalar, Device>(total_params);
    }
    if (!backuped_gradients_ || backuped_gradients_.numel() != total_params) {
      backuped_gradients_ = make_buffer<Scalar, Device>(total_params);
    }

    problem.for_each_parameter([this](auto& param) {
      auto& value = param.value();
      auto offset = param.offset();
      auto size = value.size();
      copy(backuped_parameters_.view().sub(offset, offset + size), value);
    });
    copy(backuped_gradients_.view(), problem.fused_gradients());
  }

  /// @brief Perform a step: x' <- x - step_size * neg_search_dir
  template <typename ProblemDerived, typename BlasDerived>
  void step(Scalar step_size, basic_problem<ProblemDerived, Scalar, Device>& problem,
            blas::basic_blas<BlasDerived, Scalar, Device>& bl) {
    problem.for_each_parameter([&](auto& param) {
      auto& value = param.value();
      auto offset = param.offset();
      auto size = value.size();
      auto neg_dir = neg_search_dir_.sub(offset, offset + size);
      bl.axpy(-step_size, neg_dir, value);
    });
  }

  const_view neg_search_dir_;  ///< negative search direction

  // Minimum and maximum step size relative to the input step size.
  Scalar min_rel_step_{1e-5};
  Scalar max_rel_step_{1e+2};

  // current step size
  Scalar step_size_{1};

  // backuped buffers.
  contiguous_vector_buffer<Scalar, Device> backuped_parameters_;
  contiguous_vector_buffer<Scalar, Device> backuped_gradients_;
};

template <typename Scalar, typename Device, typename Blas>
class no_linesearcher : public basic_linesearcher<no_linesearcher<Scalar, Device, Blas>, Scalar, Device> {
public:
  using base = basic_linesearcher<no_linesearcher<Scalar, Device, Blas>, Scalar, Device>;
  friend base;

  using stopping_criteria_type = typename base::stopping_criteria_type;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using result_type = typename base::result_type;
  no_linesearcher() = default;

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& /* callback */) {
    // No linesearcher, just perform a step.
    base::step(base::step_size_, problem, blas_);
    return {problem.current_value(), 0};
  }
  Blas blas_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const optim_result<Scalar>& result) {
  os << "Iter[" << result.iterations_ << "]: loss=" << result.value_ << ", |g|=" << result.grad_norm_
     << ", delt=" << result.last_change_ << ", converged=" << result.converged_;
  return os;
}

}  // namespace mathprim::optim
