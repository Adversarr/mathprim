#pragma once
#include "mathprim/optim/basic_optim.hpp"
namespace mathprim::optim {

namespace internal {
template <typename Scalar>
bool satisfies_armijo(Scalar f0,               // original energy
                      Scalar f_step,           // stepped energy
                      Scalar step_size,        // step size
                      Scalar armijo_threshold  // threshold
) {
  return f_step <= f0 + step_size * armijo_threshold;
}
}

template <typename Scalar, typename Device, typename Blas>
class backtracking_linesearcher
    : public basic_linesearcher<backtracking_linesearcher<Scalar, Device, Blas>, Scalar, Device> {
public:
  using base = basic_linesearcher<backtracking_linesearcher<Scalar, Device, Blas>, Scalar, Device>;
  friend base;

  using stopping_criteria_type = typename base::stopping_criteria_type;
  using view_type = typename base::view_type;
  using const_view = typename base::const_view;
  using result_type = typename base::result_type;

  backtracking_linesearcher() = default;

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& /* callback */) {
    auto criteria = this->criteria();
    Scalar& step_size = this->step_size_;
    Scalar old_value = problem.current_value();
    result_type result;
    auto& iterations = result.iterations_;
    auto min_abs_step = step_size * this->min_rel_step_;
    bool satisfied = false;
    for (; iterations < criteria.max_iterations_ && min_abs_step < step_size; ++iterations) {
      // Step 1: Compute the new value.
      base::step(step_size, problem, blas_);
      Scalar new_value = problem.eval_value();
      if (internal::satisfies_armijo(old_value, new_value, step_size, armijo_threshold_)) {
        satisfied = true;
        break;
      }

      // Step 2: not satisfied => restore the state and shrink the step size.
      base::restore_state(problem);
      step_size *= step_shrink_factor_;
    }
    printf("Backtracking line search: iterations = %d\n", iterations);
    if (!satisfied) {
      fprintf(stderr, "Warning: backtrack break due to step too small.\n");
    }
    return result;
  }

  Blas blas_;
public:  // Hyper parameters.
  Scalar step_shrink_factor_{0.6};
  Scalar armijo_threshold_{1e-4};  // i.e. the expect descent rate.
};

}  // namespace mathprim::optim