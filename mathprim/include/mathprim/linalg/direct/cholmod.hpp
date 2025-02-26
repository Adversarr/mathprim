#pragma once
#include <cholmod.h>

#include <Eigen/CholmodSupport>

#include "mathprim/linalg/direct/direct.hpp"
#include "mathprim/linalg/direct/eigen_support.hpp"
#include "mathprim/sparse/blas/cholmod.hpp"

namespace mathprim::sparse::direct {

template <sparse_format Format>
class cholmod_chol : public basic_direct_solver<cholmod_chol<Format>, double, Format, device::cpu> {
public:
  static_assert(Format == sparse_format::csr || Format == sparse_format::csc,
                "Cholmod only supports CSR or CSC format. (Symmetric)");

  using base = basic_direct_solver<cholmod_chol<Format>, double, Format, device::cpu>;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  friend base;

  cholmod_chol() = default;
  explicit cholmod_chol(const_matrix_view mat) : base(mat_) {
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
    if (sparse_ != nullptr) {
      // cholmod_free_sparse(&sparse_, common_);
      delete sparse_;
      sparse_ = nullptr;
    }

    if (common_ != nullptr) {
      cholmod_finish(common_);
      delete common_;
      common_ = nullptr;
    }
  }

protected:
  using base::mat_;
  cholmod_sparse* sparse_{nullptr};
  cholmod_factor* factor_{nullptr};
  // Internal workspace reuse
  cholmod_dense *Ywork_ = nullptr, *Ework_ = nullptr;
  // cholmod cm
  cholmod_common* common_{nullptr};

  void analyze_impl(int supernodal = CHOLMOD_SIMPLICIAL, int final_ll = 0, int final_asis = 1) {
    reset();
    common_ = new cholmod_common;
    cholmod_start(common_);
    // LDLT:
    common_->supernodal = supernodal;
    common_->final_asis = final_asis;
    common_->final_ll = final_ll;
    int int_true = static_cast<int>(true);

    // store the lower part.
    // {
    //   index_t lower_count = 0;
    //   auto outer = mat_.outer_ptrs();
    //   auto inner = mat_.inner_indices();
    //   auto values = mat_.values();
    //   for (index_t i = 0; i < mat_.rows(); ++i) {
    //     for (index_t j = outer[i]; j < outer[i + 1]; ++j) {
    //       if (inner[j] >= i) {
    //         lower_count++;
    //       }
    //     }
    //   }
    //   {
    //     int xtype = CHOLMOD_REAL;
    //     int dtype = CHOLMOD_DOUBLE;
    //     sparse_ = cholmod_allocate_sparse(mat_.rows(), mat_.cols(), lower_count,  // matrix
    //                                       int_true, int_true,                     // sorted, packed
    //                                       1,                                      // stype > 0 indicates symmetric
    //                                       xtype + dtype, common_);
    //   }
    //   if (sparse_ == nullptr || sparse_->stype == 0) {
    //     throw std::runtime_error("Failed to allocate sparse matrix.");
    //   }
    //   auto* row_pointer = static_cast<int32_t*>(sparse_->p);
    //   auto* col_indices = static_cast<int32_t*>(sparse_->i);
    //   auto* values_ptr = static_cast<double*>(sparse_->x);
    //   index_t current = 0;
    //   for (index_t i = 0; i < mat_.rows(); ++i) {
    //     row_pointer[i] = current;
    //     for (index_t j = outer[i]; j < outer[i + 1]; ++j) {
    //       if (inner[j] >= i) {
    //         col_indices[current] = inner[j];
    //         values_ptr[current] = values[j];
    //         current++;
    //       }
    //     }
    //   }
    //   row_pointer[mat_.rows()] = current;
    // }
    sparse_ = new cholmod_sparse;
    memset(sparse_, 0, sizeof(cholmod_sparse));
    sparse_->nrow = mat_.rows();
    sparse_->ncol = mat_.cols();
    sparse_->nzmax = mat_.nnz();
    sparse_->p = const_cast<index_t*>(mat_.outer_ptrs().data());
    sparse_->i = const_cast<index_t*>(mat_.inner_indices().data());
    sparse_->x = const_cast<double*>(mat_.values().data());
    sparse_->z = nullptr;
    sparse_->sorted = int_true;
    sparse_->packed = int_true;
    sparse_->nz = nullptr;
    sparse_->itype = CHOLMOD_INT;
    sparse_->stype = 1;
    sparse_->dtype = CHOLMOD_DOUBLE;
    sparse_->xtype = CHOLMOD_REAL;


    factor_ = cholmod_analyze(sparse_, common_);
    if (factor_ == nullptr) {
      throw std::runtime_error("Failed to analyze the matrix.");
    }
  }

  void factorize_impl() {
    double beta[2] = {0, 0};
    int fact_stat = cholmod_factorize_p(sparse_, beta, nullptr, 0, factor_, common_);
    // cholmod's return value is not reliable, we use Eigen's implementation to check it.
    if (factor_->minor != factor_->n) {
      fprintf(stderr, "Matrix is not positive definite. (retry with simplicial LDLT) %d\n", fact_stat);
      analyze_impl(CHOLMOD_SIMPLICIAL,  // supernodal
                   0,                   // final_ll
                   1);                  // final_asis
      double beta[2] = {0, 0};
      fact_stat = cholmod_factorize_p(sparse_, beta, nullptr, 0, factor_, common_);
      // cholmod_factorize(sparse_, factor_, common_);
    }

    // Second stage, if it fails, we throw an exception.
    if (factor_->minor != factor_->n) {
      throw std::runtime_error("Failed to factorize the matrix.");
    }
  }

  void solve_impl(vector_view x, const_vector y) {
    cholmod_dense rhs, lhs;
    ::std::memset(&rhs, 0, sizeof(rhs));
    ::std::memset(&lhs, 0, sizeof(lhs));
    rhs.nzmax = rhs.d = rhs.nrow = (size_t)mat_.rows();
    rhs.ncol = 1;
    rhs.x = const_cast<double*>(y.data());
    rhs.xtype = CHOLMOD_REAL;
    rhs.dtype = CHOLMOD_DOUBLE;
    lhs.nzmax = lhs.d = lhs.nrow = (size_t)mat_.rows();
    lhs.ncol = 1;
    lhs.x = x.data();
    lhs.xtype = CHOLMOD_REAL;
    lhs.dtype = CHOLMOD_DOUBLE;

    cholmod_dense* out = &lhs;
    cholmod_solve2(                         //
        CHOLMOD_A, factor_, &rhs, nullptr,  // Inputs
        &out, nullptr,                      // outputs
        &Ywork_, &Ework_, common_           // workspace
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