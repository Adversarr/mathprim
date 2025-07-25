#pragma once

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
#include "slu_ddefs.h"
#include "slu_sdefs.h"

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, sparse::sparse_format SparseCompression>
class ilu;

template <typename Scalar>
class ilu<Scalar, device::cpu, sparse::sparse_format::csr> final
    : public basic_preconditioner<ilu<Scalar, device::cpu, sparse::sparse_format::csr>, Scalar, device::cpu,
                                  sparse::sparse_format::csr> {
public:
  using base = basic_preconditioner<ilu<Scalar, device::cpu, sparse::sparse_format::csr>, Scalar, device::cpu,
                                    sparse::sparse_format::csr>;
  friend base;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = sparse::basic_sparse_view<const Scalar, device::cuda, sparse::sparse_format::csr>;
  static constexpr bool is_float32 = std::is_same_v<Scalar, float>;
  static_assert(is_float32 || std::is_same_v<Scalar, double>, "Only float32 and float64 are supported.");
  ilu() { ilu_set_default_options(&options); }

  ilu(const ilu&) = delete;
  ilu(ilu&&) = default;

  using perm_buffer = contiguous_vector_buffer<int, device::cpu>;
  using vector_buffer = contiguous_vector_buffer<Scalar, device::cpu>;

  explicit ilu(const_sparse view) : base(view) {
    ilu_set_default_options(&options);
    this->compute({});
  }

  void reset() {
    if (A_) {
      Destroy_CompCol_Matrix(A_.get());
      A_.reset();
    }
    if (L_) {
      Destroy_SuperNode_Matrix(L_.get());
      L_.reset();
    }
    if (U_) {
      Destroy_CompCol_Matrix(U_.get());
      U_.reset();
    }
    if (B_) {
      Destroy_SuperMatrix_Store(B_.get());
      B_.reset();
    }
    if (X_) {
      Destroy_SuperMatrix_Store(X_.get());
      X_.reset();
    }
    if (stat) {
      StatFree(stat.get());
      stat.reset();
    }
  }

  void factorize_impl() {
    ilu_set_default_options(&options);
    // TODO: Here we assume the input matrix is symmetric, csr is equivalent to csc
    index_t n = this->matrix().rows();
    index_t nnz = this->matrix().nnz();
    auto* outer_ptrs = const_cast<index_t*>(this->matrix().outer_ptrs().data());
    auto* inner_indices = const_cast<index_t*>(this->matrix().inner_indices().data());
    auto* values = const_cast<Scalar*>(this->matrix().values().data());
    A_ = std::make_unique<SuperMatrix>();
    L_ = std::make_unique<SuperMatrix>();
    U_ = std::make_unique<SuperMatrix>();
    B_ = std::make_unique<SuperMatrix>();
    X_ = std::make_unique<SuperMatrix>();
    // options.PivotGrowth = YES;	  /* Compute reciprocal pivot growth */
    // options.ConditionNumber = YES;/* Compute reciprocal condition number */
    options.ILU_MILU = SILU; /* Use ILU(0) */

    rhsb = make_buffer<Scalar>(n);
    rhsx = make_buffer<Scalar>(n);

    if constexpr (is_float32) {
      sCreate_CompCol_Matrix(A_.get(), n, n, nnz, values, inner_indices, outer_ptrs, SLU_NC, SLU_S, SLU_GE);
      sCreate_Dense_Matrix(B_.get(), n, 1, rhsb.data(), n, SLU_DN, SLU_S, SLU_GE);
      sCreate_Dense_Matrix(X_.get(), n, 1, rhsx.data(), n, SLU_DN, SLU_S, SLU_GE);
    } else {
      dCreate_CompCol_Matrix(A_.get(), n, n, nnz, values, inner_indices, outer_ptrs, SLU_NC, SLU_D, SLU_GE);
      dCreate_Dense_Matrix(B_.get(), n, 1, rhsb.data(), n, SLU_DN, SLU_D, SLU_GE);
      dCreate_Dense_Matrix(X_.get(), n, 1, rhsx.data(), n, SLU_DN, SLU_D, SLU_GE);
    }

    perm_r_ = make_buffer<int>(n);
    perm_c_ = make_buffer<int>(n);
    etree_ = make_buffer<int>(n);
    R_ = make_buffer<Scalar>(n);
    C_ = make_buffer<Scalar>(n);
    stat = std::make_unique<SuperLUStat_t>();
    StatInit(stat.get());
    int info;

    Scalar* work = nullptr;
    int lwork = 0;
    auto *R = R_.data(), *C = C_.data();
    auto *perm_r = perm_r_.data(), *perm_c = perm_c_.data(), *etree = etree_.data();
    Scalar rpg, rcond;

    char equed[1] = {'N'};
    B_->ncol = 0; /* not to perform triangular solution */
    if constexpr (is_float32) {
      sgsisx(&options, A_.get(), perm_c, perm_r, etree, equed, R, C, L_.get(), U_.get(), work,  //
             lwork, B_.get(), X_.get(), &rpg, &rcond, &Glu, &mem_usage, stat.get(), &info);
    } else {
      dgsisx(&options, A_.get(), perm_c, perm_r, etree, equed, R, C, L_.get(), U_.get(), work,  //
             lwork, B_.get(), X_.get(), &rpg, &rcond, &Glu, &mem_usage, stat.get(), &info);
    }
  }

  void apply_impl(vector_type y, const_vector x) {

    /* Set the options to do solve-only. */
    options.Fact = FACTORED;
    options.PivotGrowth = NO;
    options.ConditionNumber = NO;

    DNformat X, Y;
    constexpr Dtype_t type = is_float32 ? SLU_S : SLU_D;
    SuperMatrix XX = {SLU_DN, type, SLU_GE, 1, 1, &X};
    SuperMatrix YY = {SLU_DN, type, SLU_GE, 1, 1, &Y};
    char equed[1] = {'N'};
    XX.nrow = YY.nrow = this->matrix().rows();
    X.lda = Y.lda = this->matrix().rows();
    X.nzval = const_cast<Scalar*>(x.data());
    Y.nzval = y.data();

    auto* A = A_.get();
    auto* L = L_.get();
    auto* U = U_.get();
    auto* perm_r = perm_r_.data();
    auto* perm_c = perm_c_.data();
    auto* R = R_.data();
    auto* C = C_.data();
    Scalar rpg, rcond;
    int info;

    if constexpr (is_float32) {
      sgsisx(&options, A, perm_c, perm_r, NULL, equed, R, C,  //
             L, U, NULL, 0, &YY, &XX, &rpg, &rcond, NULL,//
             &mem_usage, stat.get(), &info);
    } else{
      dgsisx(&options, A, perm_c, perm_r, NULL, equed, R, C,  //
             L, U, NULL, 0, &YY, &XX, &rpg, &rcond, NULL,      //
             &mem_usage, stat.get(), &info);
    }
  }

  mem_usage_t mem_usage;
  std::unique_ptr<SuperMatrix> A_;
  std::unique_ptr<SuperMatrix> L_, U_;
  std::unique_ptr<SuperMatrix> B_, X_;
  superlu_options_t options;
  GlobalLU_t Glu;
  perm_buffer perm_r_, perm_c_, etree_;
  std::unique_ptr<SuperLUStat_t> stat;
  vector_buffer R_, C_;
  vector_buffer rhsb, rhsx;
};

}  // namespace mathprim::sparse::iterative
