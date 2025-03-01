#pragma once
#include <algorithm>
#include <vector>
#include <ostream>
#include "mathprim/blas/blas.hpp"
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/view.hpp"

namespace mathprim::optim {

template <typename Scalar, typename Device>
class parameter_item {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  using view_type = contiguous_view<Scalar, dshape<1>, Device>;
  using const_buffer_type = contiguous_view<const Scalar, dshape<1>, Device>;

  parameter_item() = default;
  parameter_item(const parameter_item&) = default;
  parameter_item(parameter_item&&) noexcept = default;
  parameter_item& operator=(const parameter_item&) = default;
  parameter_item& operator=(parameter_item&&) noexcept = default;

  parameter_item(view_type value, view_type gradient, std::string name = "") :
      value_(value), gradient_(gradient), name_(name) {}

  parameter_item(view_type value, std::string name = "") :  // NOLINT(google-explicit-constructor)
      value_(value), name_(name) {}

  const view_type& value() const noexcept { return value_; }
  const view_type& gradient() const noexcept { return gradient_; }
  const std::string& name() const noexcept { return name_; }
  index_t offset() const noexcept { return offset_; }

  template <typename Derived, typename Scalar2, typename Device2>
  friend class basic_problem;

private:
  void set_gradient(view_type gradient) noexcept { gradient_ = gradient; }
  void set_offset(index_t offset) noexcept { offset_ = offset; }

  view_type value_;
  view_type gradient_;
  index_t offset_{};
  // name of the data item (optional)
  std::string name_;
};

template <typename Derived, typename Scalar, typename Device>
class basic_problem {
public:
  using view_type = contiguous_view<Scalar, dshape<1>, Device>;
  using const_view = contiguous_view<const Scalar, dshape<1>, Device>;
  using buffer_type = contiguous_buffer<Scalar, dshape<1>, Device>;
  using parameter = parameter_item<Scalar, Device>;
  using parameter_container = std::vector<parameter>;

  basic_problem() = default;
  basic_problem(basic_problem&&) noexcept = default;
  basic_problem& operator=(basic_problem&&) noexcept = default;
  basic_problem(const basic_problem&) = default;

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  MATHPRIM_NOINLINE Scalar eval_value() {
    loss_ = 0;
    derived().eval_value_impl();
    return loss_;
  }

  MATHPRIM_NOINLINE void eval_gradients() {
    zero_gradients();
    derived().eval_gradients_impl();
  }

  // Fused evaluation of value and gradients
  MATHPRIM_NOINLINE Scalar eval_value_and_gradients() {
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
  void on_setup() {}


  void accumulate_loss(Scalar item) noexcept { loss_ += item; }

  // Reset the gradients of the parameters
  void zero_gradients() { fused_gradients_.fill_bytes(0); }

  // creates the fused gradients buffer, and set the gradients of the parameters
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

  index_t register_parameter(view_type value, std::string name = "") noexcept {
    parameters_.emplace_back(value, name);
    return static_cast<index_t>(parameters_.size() - 1);
  }

  void eval_value_and_gradients_impl() {
    derived().eval_value_impl();
    derived().eval_gradients_impl();
  }

  // void eval_gradients_impl() {}
  // void eval_value_impl() {}
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
};

template <typename Scalar>
struct stopping_criteria {
  Scalar tol_change_{1e-6};  // |f[x] - f[x_prev]| < tol_change => stop
  Scalar tol_grad_{1e-3};    // |g| < tol_grad => stop
  int max_iterations_{100};  // maximum number of iterations
};

template <typename Derived, typename Scalar, typename Device>  // Implementation of the optimizer
class basic_optimizer {
public:
  using view_type = contiguous_view<Scalar, dshape<1>, Device>;
  using const_view = contiguous_view<const Scalar, dshape<1>, Device>;

  basic_optimizer() = default;
  basic_optimizer(basic_optimizer&&) noexcept = default;
  basic_optimizer& operator=(basic_optimizer&&) noexcept = default;
  basic_optimizer(const basic_optimizer&) = default;
  using stopping_criteria_type = stopping_criteria<Scalar>;
  using result_type = optim_result<Scalar>;

  struct do_nothing_cb {
    inline void operator()(const result_type& ) {}
  };

  template <typename ProblemDerived, typename Callback = do_nothing_cb>
  result_type optimize(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback = {}) {
    return static_cast<Derived&>(*this).template optimize_impl<ProblemDerived, Callback>(
        problem, std::forward<Callback>(callback));
  }

  stopping_criteria_type &criteria() noexcept { return stopping_criteria_; }

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

  /**
   * @brief Search a step size along the search direction.
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

    neg_search_dir_ = neg_search_dir;
    step_size_ = init_step;
    auto result = base::template optimize<ProblemDerived, LinesearchCallback>(
        problem, std::forward<LinesearchCallback>(callback));

    restore_state(problem, true);
    return {result, step_size_};
  }

protected:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& callback) {
    return static_cast<Derived&>(*this).template optimize_impl<ProblemDerived, Callback>(
        problem, std::forward<Callback>(callback));
  }

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
      auto& gradient = param.gradient();
      auto offset = param.offset();
      auto size = value.size();
      copy(backuped_parameters_.view().sub(offset, offset + size), value);
    });
    copy(backuped_gradients_.view(), problem.fused_gradients());
  }

  template <typename ProblemDerived>
  void restore_state(basic_problem<ProblemDerived, Scalar, Device>& problem, bool with_grad = false) {
    problem.for_each_parameter([this](auto& param) {
      auto& value = param.value();
      auto& gradient = param.gradient();
      auto offset = param.offset();
      auto size = value.size();
      copy(value, backuped_parameters_.view().sub(offset, offset + size));
    });
    if (with_grad) {
      copy(problem.fused_gradients(), backuped_gradients_);
    }
  }

  template <typename ProblemDerived, typename BlasDerived>
  void step(Scalar step_size, basic_problem<ProblemDerived, Scalar, Device>& problem,
            blas::basic_blas<BlasDerived, Scalar, Device>& bl) {
    problem.for_each_parameter([&](auto& param) {
      auto& value = param.value();
      auto& gradient = param.gradient();
      auto offset = param.offset();
      auto size = value.size();
      auto neg_dir = neg_search_dir_.sub(offset, offset + size);
      bl.axpy(-step_size, neg_dir, value);
    });
  }

  const_view neg_search_dir_;
  Scalar min_rel_step_{1e-5}; // relative to input step_size.
  Scalar max_rel_step_{1e+2}; // relative to input step_size.
  Scalar step_size_{1};

  contiguous_vector_buffer<Scalar, Device> backuped_parameters_;
  contiguous_vector_buffer<Scalar, Device> backuped_gradients_;
};

template <typename Scalar, typename Device>
class no_linesearcher : public basic_linesearcher<no_linesearcher<Scalar, Device>, Scalar, Device> {
public:
  using base = basic_linesearcher<no_linesearcher<Scalar, Device>, Scalar, Device>;
  friend base;

  using stopping_criteria_type = typename base::stopping_criteria_type;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using result_type = typename base::result_type;
  no_linesearcher() = default;

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& /* callback */) {
    return {problem.current_value(), 0};
  }
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const optim_result<Scalar>& result) {
  os << "Iter[" << result.iterations_ << "]: loss=" << result.value_ << ", |g|=" << result.grad_norm_
      << ", delt=" << result.last_change_;
  return os;
}

}  // namespace mathprim::optim