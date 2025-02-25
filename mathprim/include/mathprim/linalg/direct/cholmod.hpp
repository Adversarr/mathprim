#pragma once
#include <cholmod.h>

#include "mathprim/linalg/direct/direct.hpp"
#include "mathprim/sparse/blas/cholmod.hpp"

namespace mathprim::sparse::direct {

template <typename Scalar, typename Device>
class cholmod_chol : public basic_direct_solver<cholmod_chol<Scalar, Device>, Scalar, sparse_format::csr, Device> {
public:
  using base = basic_direct_solver<cholmod_chol<Scalar, Device>, Scalar, sparse_format::csr, Device>;
  using vector_view = typename base::vector_view;
  using const_vector = typename base::const_vector;
  using matrix_view = typename base::matrix_view;
  using const_matrix_view = typename base::const_matrix_view;
  friend base;

  cholmod_chol() = default;
  explicit cholmod_chol(const_matrix_view mat): base(mat_) {
    base::compute(mat);
  }

  cholmod_chol(cholmod_chol&& other): base(other.mat_) {
    std::swap(sparse_, other.sparse_);
    std::swap(factor_, other.factor_);
    std::swap(Ywork_, other.Ywork_);
    std::swap(Ework_, other.Ework_);
  }

  ~cholmod_chol() {
    reset();
  }

private:
  void reset() {
    auto* common = blas::get_cholmod_handle();
    if (factor_ != nullptr) {
      cholmod_free_factor(&factor_, common);
      factor_ = nullptr;
    }
    if (Ywork_ != nullptr) {
      cholmod_free_dense(&Ywork_, common);
      Ywork_ = nullptr;
    }
    if (Ework_ != nullptr) {
      cholmod_free_dense(&Ework_, common);
      Ework_ = nullptr;
    }
    if (sparse_ != nullptr) {
      cholmod_free_sparse(&sparse_, common);
      sparse_ = nullptr;
    }
  }

protected:
  using base::mat_;
  cholmod_sparse* sparse_ = nullptr;
  cholmod_factor *factor_{nullptr};
  // Internal workspace reuse
  cholmod_dense *Ywork_ = nullptr, *Ework_ = nullptr;


  void analyze_impl() {
    auto* common = blas::get_cholmod_handle();
    reset();
    // store the lower part.
    index_t lower_count = 0;
    auto outer = mat_.outer_ptrs();
    auto inner = mat_.inner_indices();
    auto values = mat_.values();
    for (index_t i = 0; i < mat_.rows(); ++i) {
      for (index_t j = outer[i]; j < outer[i + 1]; ++j) {
        if (inner[j] <= i) {
          lower_count++;
        }
      }
    }
    int int_true = static_cast<int>(true);
    {
      int xtype = CHOLMOD_REAL;
      int dtype = std::is_same_v<Scalar, float> ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
      sparse_ = cholmod_allocate_sparse(mat_.rows(), mat_.cols(), lower_count,  // matrix
                                        int_true, int_true,                     // sorted, packed
                                        1,                                      // stype > 0 indicates symmetric
                                        xtype + dtype, common);
    }
    if (sparse_ == nullptr || sparse_->stype == 0) {
      throw std::runtime_error("Failed to allocate sparse matrix.");
    }

    auto* row_pointer = static_cast<int32_t*>(sparse_->p);
    auto* col_indices = static_cast<int32_t*>(sparse_->i);
    auto* values_ptr = static_cast<Scalar*>(sparse_->x);
    index_t current = 0;
    for (index_t i = 0; i < mat_.rows(); ++i) {
      row_pointer[i] = current;
      for (index_t j = outer[i]; j < outer[i + 1]; ++j) {
        if (inner[j] <= i) {
          col_indices[current] = inner[j];
          values_ptr[current] = values[j];
          current++;
        }
      }
    }
    row_pointer[mat_.rows()] = current;

    factor_ = cholmod_analyze(sparse_, common);
    if (factor_ == nullptr) {
      throw std::runtime_error("Failed to analyze the matrix.");
    }
  }

  void factorize_impl() {
    auto* common = blas::get_cholmod_handle();
    cholmod_factorize(sparse_, factor_, common);
    // cholmod's return value is not reliable, we use Eigen's implementation to check it.
    if (factor_->minor != factor_->n) {
      fprintf(stderr, "Matrix is not positive definite. (retry with simplicial)\n");
      auto previous_method = common->supernodal;
      common->supernodal = CHOLMOD_SIMPLICIAL;
      try {
        analyze_impl();
      } catch (...) {
        common->supernodal = previous_method;
        throw;
      }
      cholmod_factorize(sparse_, factor_, common);
      common->supernodal = previous_method; // restore it back anyway.
    }

    // Second stage, if it fails, we throw an exception.
    if (factor_->minor != factor_->n) {
      cholmod_print_sparse(sparse_, "A", common);
      cholmod_print_factor(factor_, "L", common);
      auto out = cholmod_check_sparse(sparse_, common);
      fprintf(stderr, "Check sparse result: %s\n", blas::internal::to_string(out));
      throw std::runtime_error("Failed to factorize the matrix.");
    }
  }

  void solve_impl(vector_view x, const_vector y) {
    auto* common = blas::get_cholmod_handle();
    cholmod_dense rhs, lhs;
    ::std::memset(&rhs, 0, sizeof(rhs));
    ::std::memset(&lhs, 0, sizeof(lhs));
    rhs.nzmax = rhs.d = rhs.nrow = (size_t) mat_.rows();
    rhs.ncol = 1;
    rhs.x = const_cast<Scalar*>(y.data());
    rhs.xtype = CHOLMOD_REAL;
    rhs.dtype = std::is_same_v<Scalar, float> ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
    lhs.nzmax = lhs.d = lhs.nrow = (size_t) mat_.rows();
    lhs.ncol = 1;
    lhs.x = x.data();
    lhs.xtype = CHOLMOD_REAL;
    lhs.dtype = std::is_same_v<Scalar, float> ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
    
    cholmod_dense* out = &lhs;
    cholmod_solve2(            //
        CHOLMOD_A, factor_, &rhs, nullptr,  // Inputs
        &out, nullptr,                      // outputs
        &Ywork_, &Ework_, common            // workspace
    );
  }
};

}