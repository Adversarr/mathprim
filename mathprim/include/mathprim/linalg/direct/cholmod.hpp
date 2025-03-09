#pragma once
#include <cholmod.h>

#include <Eigen/CholmodSupport>

#include "mathprim/linalg/direct/direct.hpp"
#include "mathprim/linalg/direct/eigen_support.hpp"
#include "mathprim/sparse/blas/cholmod.hpp"

namespace mathprim::sparse::direct {

template <typename Scalar, sparse_format Format>
class cholmod_chol : public basic_direct_solver<cholmod_chol<Scalar, Format>, Scalar, Format, device::cpu> {
public:
  static_assert(Format == sparse_format::csr || Format == sparse_format::csc,
                "Cholmod only supports CSR or CSC format. (Symmetric)");

  using base = basic_direct_solver<cholmod_chol<Scalar, Format>, Scalar, Format, device::cpu>;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using sparse_view = typename base::sparse_view;
  using const_sparse = typename base::const_sparse;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  friend base;

  cholmod_chol() = default;
  explicit cholmod_chol(const_sparse mat) : base(mat) {
    base::compute(mat);
  }

  cholmod_chol(cholmod_chol&& other) : base(other.mat_) {
    std::swap(sparse_, other.sparse_);
    std::swap(factor_, other.factor_);
    std::swap(Ywork_, other.Ywork_);
    std::swap(Ework_, other.Ework_);
    std::swap(common_, other.common_);
  }

  ~cholmod_chol() {
    reset();
  }

private:
  void reset() {
    if (factor_ != nullptr) {
      cholmod_free_factor(&factor_, common_);
      factor_ = nullptr;
    }
    if (Ywork_ != nullptr) {
      cholmod_free_dense(&Ywork_, common_);
      Ywork_ = nullptr;
    }
    if (Ework_ != nullptr) {
      cholmod_free_dense(&Ework_, common_);
      Ework_ = nullptr;
    }

    if (common_ != nullptr) {
      cholmod_finish(common_);
      delete common_;
      common_ = nullptr;
    }
  }

protected:
  using base::mat_;
  cholmod_sparse sparse_{};
  cholmod_factor* factor_{nullptr};
  // Internal workspace reuse
  cholmod_dense *Ywork_ = nullptr, *Ework_ = nullptr;
  // cholmod cm
  cholmod_common* common_{nullptr};

  int dtype() const {
    if constexpr (std::is_same_v<Scalar, float>) {
      return CHOLMOD_SINGLE;
    } else if constexpr (std::is_same_v<Scalar, double>) {
      return CHOLMOD_DOUBLE;
    } else {
      static_assert(internal::always_false_v<Scalar>, "Unsupported scalar type.");
    }
  }

  void analyze_impl(int supernodal = CHOLMOD_SUPERNODAL, int final_asis = 0, int final_ll = 1) {
    const int int_true = static_cast<int>(true);
    reset();
    common_ = new cholmod_common;
    cholmod_start(common_);
    common_->supernodal = supernodal;
    common_->final_asis = final_asis;
    common_->final_ll = final_ll;
    memset(&sparse_, 0, sizeof(cholmod_sparse));

    sparse_.nrow = mat_.rows();
    sparse_.ncol = mat_.cols();
    sparse_.nzmax = mat_.nnz();
    sparse_.p = const_cast<index_t*>(mat_.outer_ptrs().data());
    sparse_.i = const_cast<index_t*>(mat_.inner_indices().data());
    sparse_.x = const_cast<Scalar*>(mat_.values().data());
    sparse_.z = nullptr;
    sparse_.sorted = int_true;
    sparse_.packed = int_true;
    sparse_.nz = nullptr;
    if constexpr (sizeof(index_t) == 4) {
      sparse_.itype = CHOLMOD_INT;
    } else if constexpr (sizeof(index_t) == 8) {
      sparse_.itype = CHOLMOD_LONG;
    } else {
      static_assert(internal::always_false_v<Scalar>, "Unsupported index type.");
    }
    sparse_.stype = -1;
    sparse_.dtype = dtype();
    sparse_.xtype = CHOLMOD_REAL;

    factor_ = cholmod_analyze(&sparse_, common_);
    if (factor_ == nullptr) {
      throw std::runtime_error("Failed to analyze the matrix.");
    }
  }

  void factorize_impl() {
    double beta[2] = {0, 0};
    int fact_stat = cholmod_factorize_p(&sparse_, beta, nullptr, 0, factor_, common_);
    // First stage, if it fails, retry.
    if (factor_->minor != factor_->n /* || fact_stat != CHOLMOD_OK */) {
      fprintf(stderr, "WARN: CHOLMOD+Supernodal to factorize the matrix. %d\n", fact_stat);
      analyze_impl(CHOLMOD_SIMPLICIAL, 1, 0); // Simplicial LDLT.
      fact_stat = cholmod_factorize_p(&sparse_, beta, nullptr, 0, factor_, common_);
    }

    // Second stage, if it fails, we throw an exception.
    if (factor_->minor != factor_->n /* || fact_stat != CHOLMOD_OK */) {
      throw std::runtime_error("Failed to factorize the matrix." + std::string(blas::internal::to_string(fact_stat)));
    }
  }

  void solve_impl(vector_view x, const_vector y) {
    cholmod_dense rhs, lhs;
    ::std::memset(&rhs, 0, sizeof(rhs));
    ::std::memset(&lhs, 0, sizeof(lhs));
    rhs.nzmax = rhs.d = rhs.nrow = static_cast<size_t>(mat_.rows());
    rhs.ncol = 1;
    rhs.x = const_cast<Scalar*>(y.data());
    rhs.xtype = CHOLMOD_REAL;
    rhs.dtype = dtype();
    lhs.nzmax = lhs.d = lhs.nrow = static_cast<size_t>(mat_.rows());
    lhs.ncol = 1;
    lhs.x = x.data();
    lhs.xtype = CHOLMOD_REAL;
    lhs.dtype = dtype();

    cholmod_dense* out = &lhs;
    cholmod_solve2(                         //
        CHOLMOD_A, factor_, &rhs, nullptr,  // Inputs
        &out, nullptr,                      // outputs
        &Ywork_, &Ework_, common_           // workspace
    );
  }

  void vsolve_impl(matrix_view x, const_matrix_view y) {
    cholmod_dense rhs, lhs;
    ::std::memset(&rhs, 0, sizeof(rhs));
    ::std::memset(&lhs, 0, sizeof(lhs));
    rhs.d = rhs.nrow = static_cast<size_t>(mat_.rows());
    rhs.nzmax = y.numel();
    rhs.ncol = static_cast<size_t>(y.shape(1));
    rhs.x = const_cast<Scalar*>(y.data());
    rhs.xtype = CHOLMOD_REAL;
    rhs.dtype = dtype();
    lhs.d = lhs.nrow = static_cast<size_t>(mat_.rows());
    lhs.nzmax = x.numel();
    lhs.ncol = static_cast<size_t>(x.shape(1));
    lhs.x = x.data();
    lhs.xtype = CHOLMOD_REAL;
    lhs.dtype = dtype();

    cholmod_dense* out = &lhs;
    cholmod_solve2(                          //
        CHOLMOD_A, factor_, &rhs, nullptr,  // Inputs
        &out, nullptr,                       // outputs
        &Ywork_, &Ework_, common_            // workspace
    );
  }
};

template <sparse::sparse_format Format>
using eigen_cholmod_simplicial_ldlt = basic_eigen_direct_solver<
    Eigen::CholmodSimplicialLDLT<eigen_support::internal::eigen_sparse_format_t<double, Format>>, double, Format>;

template <sparse::sparse_format Format>
using eigen_cholmod_simplicial_llt = basic_eigen_direct_solver<
    Eigen::CholmodSimplicialLLT<eigen_support::internal::eigen_sparse_format_t<double, Format>>, double, Format>;

// Super node version.
template <sparse::sparse_format Format>
using eigen_cholmod_supernodal_llt = basic_eigen_direct_solver<
    Eigen::CholmodSupernodalLLT<eigen_support::internal::eigen_sparse_format_t<double, Format>>, double, Format>;

}  // namespace mathprim::sparse::direct