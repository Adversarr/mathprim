#pragma once
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"  // map for sparse matrix

namespace mathprim::sparse::iterative {

template <typename Scalar, typename EigenPreconditioner>
class eigen_preconditioner
    : public basic_preconditioner<eigen_preconditioner<Scalar, EigenPreconditioner>, Scalar, device::cpu> {
public:
  using base_type = basic_preconditioner<eigen_preconditioner<Scalar, EigenPreconditioner>, Scalar, device::cpu>;
  using vector_type = typename base_type::vector_type;
  using const_vector = typename base_type::const_vector;
  friend base_type;

  eigen_preconditioner() = default;
  template <typename MatType>
  explicit eigen_preconditioner(const MatType& mat) {
    compute(mat);
  }

  eigen_preconditioner(eigen_preconditioner&&) = default;

  EigenPreconditioner& impl() {
    MATHPRIM_INTERNAL_CHECK_THROW(impl_, std::runtime_error, "The preconditioner is not initialized.");
    return *impl_;
  }

  const EigenPreconditioner& impl() const {
    MATHPRIM_INTERNAL_CHECK_THROW(impl_, std::runtime_error, "The preconditioner is not initialized.");
    return impl_;
  }

  template <typename MatType>
  MATHPRIM_NOINLINE void compute(const MatType& mat) {
    auto mat_map = eigen_support::map(mat).eval();
    if (!impl_) {
      impl_ = std::make_unique<EigenPreconditioner>(mat_map);
    } else {
      impl_->compute(mat_map);
    }

    if (impl().info() != Eigen::Success) {
      throw std::runtime_error("Eigen preconditioner failed to compute: " + eigen_support::to_string(impl().info()));
    }
  }

protected:
  void apply_impl(vector_type y, const_vector x) {
    auto map_y = eigen_support::cmap(y);
    auto map_x = eigen_support::cmap(x);
    map_y = impl().solve(map_x);
  }

  std::unique_ptr<EigenPreconditioner> impl_;
};

template <typename Scalar>
using eigen_ichol = eigen_preconditioner<Scalar, Eigen::IncompleteCholesky<Scalar>>;

template <typename Scalar>
using eigen_ilu = eigen_preconditioner<Scalar, Eigen::IncompleteLUT<Scalar>>;

///! Use diagonal_preconditioner instead
template <typename Scalar>
using eigen_diagonal_preconditioner = eigen_preconditioner<Scalar, Eigen::DiagonalPreconditioner<Scalar>>;

///! Use none_preconditioner instead
template <typename Scalar>
using eigen_identity_preconditioner = eigen_preconditioner<Scalar, Eigen::IdentityPreconditioner>;

}  // namespace mathprim::sparse::iterative