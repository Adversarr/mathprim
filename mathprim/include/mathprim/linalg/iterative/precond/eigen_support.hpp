#pragma once
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/supports/eigen_dense.hpp"
#include "mathprim/supports/eigen_sparse.hpp"  // map for sparse matrix

namespace mathprim::sparse::iterative {

template <typename Scalar, typename EigenPreconditioner, sparse_format Compression>
class eigen_preconditioner final
    : public basic_preconditioner<eigen_preconditioner<Scalar, EigenPreconditioner, Compression>, Scalar, device::cpu,
                                  Compression> {
public:
  using base = basic_preconditioner<eigen_preconditioner<Scalar, EigenPreconditioner, Compression>, Scalar, device::cpu,
                                    Compression>;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = typename base::const_sparse;

  friend base;

  eigen_preconditioner() = default;
  explicit eigen_preconditioner(const_sparse mat) : base(mat) { this->compute({}); }

  eigen_preconditioner(eigen_preconditioner&&) = default;

  EigenPreconditioner& impl() {
    MATHPRIM_INTERNAL_CHECK_THROW(impl_, std::runtime_error, "The preconditioner is not initialized.");
    return *impl_;
  }

  const EigenPreconditioner& impl() const {
    MATHPRIM_INTERNAL_CHECK_THROW(impl_, std::runtime_error, "The preconditioner is not initialized.");
    return impl_;
  }

protected:
  void analyze_impl() {
    auto mat = this->matrix();
    auto mat_map = eigen_support::map(mat).eval();
    impl_ = std::make_unique<EigenPreconditioner>();
    impl_->analyzePattern(mat_map);
  }

  void factorize_impl() {
    auto mat = this->matrix();
    auto mat_map = eigen_support::map(mat).eval();
    impl_->factorize(mat_map);
  }

  void apply_impl(vector_type y, const_vector x) {
    auto map_y = eigen_support::cmap(y);
    auto map_x = eigen_support::cmap(x);
    map_y = impl().solve(map_x);
  }

  std::unique_ptr<EigenPreconditioner> impl_;
};

template <typename Scalar, sparse_format Compression>
using eigen_ichol = eigen_preconditioner<Scalar, Eigen::IncompleteCholesky<Scalar>, Compression>;

template <typename Scalar, sparse_format Compression>
using eigen_ilu = eigen_preconditioner<Scalar, Eigen::IncompleteLUT<Scalar>, Compression>;

///! Use diagonal_preconditioner instead, this class is reserved for debug use
template <typename Scalar, sparse_format Compression>
using eigen_diagonal_preconditioner = eigen_preconditioner<Scalar, Eigen::DiagonalPreconditioner<Scalar>, Compression>;

///! Use none_preconditioner instead, this class is reserved for debug use
template <typename Scalar, sparse_format Compression>
using eigen_identity_preconditioner = eigen_preconditioner<Scalar, Eigen::IdentityPreconditioner, Compression>;

}  // namespace mathprim::sparse::iterative
