#pragma once
#include <suitesparse/cholmod.h>

#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/utils/singleton.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::singletons {

class cholmod_handle : public internal::basic_singleton<cholmod_handle, cholmod_common*> {
public:
  using base = internal::basic_singleton<cholmod_handle, cholmod_common*>;
  friend base;

private:
  void create_impl(cholmod_common*& handle) noexcept {
    handle = new cholmod_common;
    cholmod_start(handle);
    handle->supernodal = CHOLMOD_AUTO;
  }

  void destroy_impl(cholmod_common*& handle) noexcept {
    cholmod_finish(handle);
    delete handle;
  }
};

}  // namespace mathprim::singletons

namespace mathprim::sparse {
namespace blas {

inline cholmod_common* get_cholmod_handle() {
  return singletons::cholmod_handle::get();
}

namespace internal {

inline const char* to_string(int status) {
  // cholmod_status -> string
  if (status == CHOLMOD_OK) {
    return "CHOLMOD_OK";
  } else if (status == CHOLMOD_NOT_INSTALLED) {
    return "CHOLMOD_NOT_INSTALLED";
  } else if (status == CHOLMOD_NOT_POSDEF) {
    return "CHOLMOD_NOT_POSDEF";
  } else if (status == CHOLMOD_OUT_OF_MEMORY) {
    return "CHOLMOD_OUT_OF_MEMORY";
  } else if (status == CHOLMOD_TOO_LARGE) {
    return "CHOLMOD_TOO_LARGE";
  } else if (status == CHOLMOD_INVALID) {
    return "CHOLMOD_INVALID";
  } else if (status == CHOLMOD_OK) {
    return "CHOLMOD_OK";
  } else {
    return "CHOLMOD_UNKNOWN";
  }
}

}

template <typename Scalar, sparse_format SparseCompression>
class cholmod : public sparse_blas_base<cholmod<Scalar, SparseCompression>, Scalar, device::cpu, SparseCompression> {
public:
  using base = sparse_blas_base<cholmod<Scalar, SparseCompression>, Scalar, device::cpu, SparseCompression>;
  friend base;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  using base::base;

  explicit cholmod(const_sparse_view mat);

private:
  void gemv_impl(Scalar alpha, const_vector_view x, Scalar beta, vector_view y, bool transpose);
  cholmod_sparse chol_mat_;
  cholmod_dense chol_x_;
  cholmod_dense chol_y_;
};
}  // namespace blas

template <typename Scalar, sparse_format SparseCompression>
blas::cholmod<Scalar, SparseCompression>::cholmod(const_sparse_view mat) : base(mat) {
  cholmod_common* cholmod_common = get_cholmod_handle();
  MATHPRIM_UNUSED(cholmod_common); // We keep it here to init the handle.
  
  ::std::memset(&chol_mat_, 0, sizeof(chol_mat_));
  if constexpr (SparseCompression == sparse_format::csc) {
    chol_mat_.nrow = mat.cols();
    chol_mat_.ncol = mat.rows();
  } else if constexpr (SparseCompression == sparse_format::csr) {
    chol_mat_.nrow = mat.rows();
    chol_mat_.ncol = mat.cols();
  } else {
    static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported sparse format for cholmod");
  }
  chol_mat_.nzmax = mat.nnz();
  chol_mat_.p = const_cast<index_t*>(mat.outer_ptrs().data());
  chol_mat_.i = const_cast<index_t*>(mat.inner_indices().data());
  chol_mat_.x = const_cast<Scalar*>(mat.values().data());
  chol_mat_.stype = 0;// lower and upper.
  chol_mat_.itype = sizeof(index_t) == 4 ? CHOLMOD_INT : CHOLMOD_LONG;
  chol_mat_.xtype = CHOLMOD_REAL;
  if constexpr (std::is_same_v<Scalar, float>) {
    chol_mat_.dtype = CHOLMOD_SINGLE;
  } else if constexpr (std::is_same_v<Scalar, double>) {
    chol_mat_.dtype = CHOLMOD_DOUBLE;
  } else {
    static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported scalar type for cholmod");
  }
  chol_mat_.sorted = 1;
  chol_mat_.packed = 1;

  // vector chol_x: (col, 1)
  ::std::memset(&chol_x_, 0, sizeof(chol_x_));
  chol_x_.nrow = mat.cols();
  chol_x_.ncol = 1;
  chol_x_.nzmax = mat.cols();
  chol_x_.d = mat.cols();
  chol_x_.x = nullptr;
  chol_x_.xtype = chol_mat_.xtype;
  chol_x_.dtype = chol_mat_.dtype;
  
  // vector chol_y: (row, 1)
  ::std::memset(&chol_y_, 0, sizeof(chol_y_));
  chol_y_.nrow = mat.rows();
  chol_y_.ncol = 1;
  chol_y_.nzmax = mat.rows();
  chol_y_.d = mat.rows();
  chol_y_.x = nullptr;
  chol_y_.xtype = chol_mat_.xtype;
  chol_y_.dtype = chol_mat_.dtype;
}

template <typename Scalar, sparse_format SparseCompression>
void blas::cholmod<Scalar, SparseCompression>::gemv_impl(Scalar alpha, const_vector_view x, Scalar beta, vector_view y,
                                                         bool transpose) {
  cholmod_common* cholmod_common = get_cholmod_handle();
  int result = 0;
  double alpha_arg = alpha;
  double beta_arg = beta;
  if constexpr (SparseCompression == sparse_format::csc) {
    // ok, suitesparse native format.
    if (!transpose) {
      // normal path, set chol_x=x, chol_y=y, and call cholmod_sdmult.
      chol_x_.x = const_cast<Scalar*>(x.data());
      chol_y_.x = y.data();

      result = cholmod_sdmult(&chol_mat_, 0, &alpha_arg, &beta_arg, &chol_x_, &chol_y_, cholmod_common);
    } else {
      // transpose path, set chol_x=y, chol_y=x, and call cholmod_sdmult.
      // y: (col, 1), x: (row, 1)
      chol_x_.x = y.data();
      chol_y_.x = const_cast<Scalar*>(x.data());

      result = cholmod_sdmult(&chol_mat_, 1, &alpha_arg, &beta_arg, &chol_y_, &chol_x_, cholmod_common);
    }
  } else if constexpr (SparseCompression == sparse_format::csr) {
    // view it as csc and do correspondingly.
    if (transpose){
      // a transpose of csr is csc, set chol_x=y, chol_y=x, and call cholmod_sdmult.
      // y: (col, 1), x: (row, 1)
      chol_x_.x = y.data();
      chol_y_.x = const_cast<Scalar*>(x.data());

      result = cholmod_sdmult(&chol_mat_, 0, &alpha_arg, &beta_arg, &chol_y_, &chol_x_, cholmod_common);
    } else {
      // csr, we transpose everything.
      // a transpose of csr is csc, set chol_x=x, chol_y=y, and call cholmod_sdmult.
      chol_x_.x = const_cast<Scalar*>(x.data());
      chol_y_.x = y.data();

      result = cholmod_sdmult(&chol_mat_, 1, &alpha_arg, &beta_arg, &chol_x_, &chol_y_, cholmod_common);
    }
  } else {
    static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported sparse format for cholmod");
  }

  if (result == 0) {
    throw std::runtime_error(std::string("Cholmod gemv failed.") + internal::to_string(cholmod_common->status));
  }
}

}  // namespace mathprim::sparse