#pragma once
#include "mathprim/optim/basic_optim.hpp"
#include "mathprim/optim/linesearcher/backtracking.hpp"
// #include "mathprim/linalg/"
namespace mathprim::optim {

template <typename Scalar, typename Device, typename Blas,
          typename Linesearcher = backtracking_linesearcher<Scalar, Device, Blas>>
          // typename SparseSolver = sparse::direct::>
class newton_optimizer : public basic_optimizer<newton_optimizer<Scalar, Device, Blas, Linesearcher>, Scalar, Device> {
public:
  using base = basic_optimizer<newton_optimizer<Scalar, Device, Blas, Linesearcher>, Scalar, Device>;
  friend base;
  using stopping_criteria_type = typename base::stopping_criteria_type;
  using result_type = typename base::result_type;

  newton_optimizer() = default;
  MATHPRIM_INTERNAL_MOVE(newton_optimizer, default);
  MATHPRIM_INTERNAL_COPY(newton_optimizer, delete);
};
}