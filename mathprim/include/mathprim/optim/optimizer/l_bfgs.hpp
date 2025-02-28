// mathprim/include/mathprim/optim/optimizer/l_bfgs.hpp
#pragma once
#include "mathprim/blas/blas.hpp"
#include "mathprim/optim/basic_optim.hpp"
#include <algorithm>
#include <cmath>

namespace mathprim::optim {

/**
 * @brief L-BFGS Optimizer.
 * @ref   https://en.wikipedia.org/wiki/Limited-memory_BFGS
 *
 * @tparam Scalar
 * @tparam Device
 * @tparam Blas
 */
template <typename Scalar, typename Device, typename Blas>
class l_bfgs_optimizer : public basic_optimizer<l_bfgs_optimizer<Scalar, Device, Blas>, Scalar, Device> {
public:
  using base = basic_optimizer<l_bfgs_optimizer<Scalar, Device, Blas>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;

  l_bfgs_optimizer() = default;

private:
  template <typename ProblemDerived, typename Callback>
  result_type optimize_impl(basic_problem<ProblemDerived, Scalar, Device>& problem, Callback&& /* callback */) {
    blas::basic_blas<Blas, Scalar, Device>& bl = blas_;
    auto criteria = base::criteria();
    auto gradients_view = problem.fused_gradients();
    result_type result;
    Scalar& value = result.value_;
    Scalar& last_change = result.last_change_;
    Scalar& grad_norm = result.grad_norm_;
    index_t& iteration = result.iterations_;

    // 1. prepare all the buffers.

    // 2. initialize the parameters.

    // 3. main loop.
  }

  contiguous_matrix_buffer<Scalar, Device> s_, y_;
  std::vector<Scalar> rho_;
  Blas blas_;

public:  // Hyper parameters.
  Scalar learning_rate_{1.0};
  int memory_size_{10};  // Number of previous steps to store in memory
};
}  // namespace mathprim::optim