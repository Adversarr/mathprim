#pragma once

#ifndef MATHPRIM_BLAS_VENDOR_INTEL_MKL
#  error "This file should only be included when MKL is enabled."
#endif

#include <mkl_spblas.h>
#include <iostream>
#include <sstream>
#include "mathprim/sparse/basic_sparse.hpp"

namespace mathprim::sparse {
namespace blas {

namespace internal {

inline const char* mkl_status_to_string(sparse_status_t status) {
  switch (status) {
    case SPARSE_STATUS_SUCCESS:
      return "SPARSE_STATUS_SUCCESS";
    case SPARSE_STATUS_NOT_INITIALIZED:
      return "SPARSE_STATUS_NOT_INITIALIZED";
    case SPARSE_STATUS_ALLOC_FAILED:
      return "SPARSE_STATUS_ALLOC_FAILED";
    case SPARSE_STATUS_INVALID_VALUE:
      return "SPARSE_STATUS_INVALID_VALUE";
    case SPARSE_STATUS_EXECUTION_FAILED:
      return "SPARSE_STATUS_EXECUTION_FAILED";
    case SPARSE_STATUS_INTERNAL_ERROR:
      return "SPARSE_STATUS_INTERNAL_ERROR";
    default:
      return "Unknown MKL status";
  }
}
}

#define MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(cond, msg)                                                           \
  do {                                                                                                          \
    sparse_status_t status = (cond);                                                                              \
    if (status != SPARSE_STATUS_SUCCESS) {                                                                      \
      std::ostringstream oss;                                                                                   \
      oss << "Check " #cond " failed, got " << ::mathprim::sparse::blas::internal::mkl_status_to_string(status) \
          << ", at " << __FILE__ << ":" << __LINE__ << "\n"                                                     \
          << msg;                                                                                               \
      throw std::runtime_error(oss.str());                                                                      \
    }                                                                                                           \
  } while (0)

#define MATHPRIM_INTERNAL_ASSERT_MKL_SPBLAS(cond, msg)                                                           \
  do {                                                                                                           \
    sparse_status_t status = (cond);                                                                             \
    if (status != SPARSE_STATUS_SUCCESS) {                                                                       \
      std::ostringstream oss;                                                                                    \
      oss << "Assert " #cond " failed, got " << ::mathprim::sparse::blas::internal::mkl_status_to_string(status) \
          << ", at " << __FILE__ << ":" << __LINE__ << "\n"                                                      \
          << msg;                                                                                                \
      std::cerr << oss.str();                                                                                    \
      std::cerr.flush();                                                                                         \
      std::abort();                                                                                              \
    }                                                                                                            \
  } while (0)

template <typename Scalar, sparse_format Compression>
class mkl : public sparse_blas_base<mkl<Scalar, Compression>, Scalar, device::cpu, Compression> {
public:
  static_assert(Compression == sparse_format::csr || Compression == sparse_format::csc,
                "MKL only supports CSR and CSC formats.");

  using base = sparse_blas_base<mkl<Scalar, Compression>, Scalar, device::cpu, Compression>;
  friend base;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;

  mkl() = default;
  explicit mkl(const_sparse_view mat);

  mkl(const mkl&) = delete;
  mkl& operator=(const mkl&) = delete;
  mkl(mkl&& other);
  mkl& operator=(mkl&&);
  ~mkl();

private:
  // y = alpha * A * x + beta * y.
  void gemv_impl(Scalar alpha, const_vector_view x, Scalar beta, vector_view y, bool transpose) {
    if constexpr (Compression == sparse_format::csr) {
      if (transpose) {  // Computes A.T @ x
        if (this->mat_.property() == sparse_property::symmetric) {
          // Symmetric matrix, use the same code path for both transposed and non-transposed.
          gemv_no_trans(alpha, x, beta, y);
        } else if (this->mat_.property() == sparse_property::skew_symmetric) {
          // A = -A.T => A.T @ x = -A @ x
          gemv_no_trans(-alpha, x, beta, y);
        } else {
          gemv_trans(alpha, x, beta, y);  // always slower sequential
        }
      } else {  // Computes A @ x
        gemv_no_trans(alpha, x, beta, y);
      }
    } else /* csc */ {
      if (transpose) {  // Computes A.T @ x
        gemv_trans(alpha, x, beta, y);
      } else {  // Computes A @ x
        if (this->mat_.property() == sparse_property::symmetric) {
          // Symmetric matrix, use the same code path for both transposed and non-transposed.
          gemv_trans(alpha, x, beta, y);
        } else if (this->mat_.property() == sparse_property::skew_symmetric) {
          // A = -A.T => A.T @ x = -A @ x
          gemv_trans(-alpha, x, beta, y);
        } else {
          gemv_no_trans(alpha, x, beta, y);
        }
      }
    }
  }

  void gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  void gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y);
  sparse_matrix_t mat_desc_ = nullptr;
  matrix_descr descr_;

  using index_buffer = contiguous_vector_buffer<MKL_INT, device::cpu>;
  index_buffer outer_ptrs_;
  index_buffer inner_indices_;
};

template <typename Scalar, sparse_format Compression>
mkl<Scalar, Compression>::mkl(const_sparse_view mat) : base(mat) {
  auto* values = const_cast<Scalar*>(this->mat_.values().data());
  const auto rows = this->mat_.rows();
  const auto cols = this->mat_.cols();
  
  // outer is int, but MKL_INT is long long. We need to copy it.
  outer_ptrs_ = make_buffer<MKL_INT>(this->mat_.outer_ptrs().size());
  inner_indices_ = make_buffer<MKL_INT>(this->mat_.inner_indices().size());
  auto* origin_outer = this->mat_.outer_ptrs().data();
  auto* origin_inner = this->mat_.inner_indices().data();
  auto* outer = outer_ptrs_.data();
  auto* inner = inner_indices_.data();
  index_t outer_size = this->mat_.outer_ptrs().size(), inner_size = this->mat_.inner_indices().size();
  for (index_t i = 0; i < outer_size; ++i) {
    outer[i] = static_cast<MKL_INT>(origin_outer[i]);
  }
  for (index_t i = 0; i < inner_size; ++i) {
    inner[i] = static_cast<MKL_INT>(origin_inner[i]);
  }
  if constexpr (Compression == sparse_format::csr) {
    if constexpr (std::is_same_v<Scalar, float>) {
      MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(                                                                            //
          mkl_sparse_s_create_csr(&mat_desc_, SPARSE_INDEX_BASE_ZERO, rows, cols, outer, outer + 1, inner, values),  //
          "mkl::mkl failed");
    } else if constexpr (std::is_same_v<Scalar, double>) {
      MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(                                                                            //
          mkl_sparse_d_create_csr(&mat_desc_, SPARSE_INDEX_BASE_ZERO, rows, cols, outer, outer + 1, inner, values),  //
          "mkl::mkl failed");
    }
  } else {
    if constexpr (std::is_same_v<Scalar, float>) {
      MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(                                                                            //
          mkl_sparse_s_create_csc(&mat_desc_, SPARSE_INDEX_BASE_ZERO, rows, cols, outer, outer + 1, inner, values),  //
          "mkl::mkl failed");
    } else if constexpr (std::is_same_v<Scalar, double>) {
      MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(                                                                            //
          mkl_sparse_d_create_csc(&mat_desc_, SPARSE_INDEX_BASE_ZERO, rows, cols, outer, outer + 1, inner, values),  //
          "mkl::mkl failed");
    }
  }

  descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr_.mode = SPARSE_FILL_MODE_FULL;
  descr_.diag = SPARSE_DIAG_NON_UNIT;
}

template <typename Scalar, sparse_format Compression>
mkl<Scalar, Compression>::mkl(mkl&& other) : base(other.mat_) {
  if (mat_desc_) {
    MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(  //
        mkl_sparse_destroy(mat_desc_),   //
        "mkl::mkl failed");
  }

  mat_desc_ = other.mat_desc_;
  descr_ = other.descr_;
  other.mat_desc_ = nullptr;

  outer_ptrs_ = std::move(other.outer_ptrs_);
  inner_indices_ = std::move(other.inner_indices_);
}

template <typename Scalar, sparse_format Compression>
mkl<Scalar, Compression>& mkl<Scalar, Compression>::operator=(mkl&& other) {
  if (this != &other) {
    if (mat_desc_) {
      MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(  //
          mkl_sparse_destroy(mat_desc_),   //
          "mkl::operator= failed");
    }
    mat_desc_ = other.mat_desc_;
    descr_ = other.descr_;
    other.mat_desc_ = nullptr;

    outer_ptrs_ = std::move(other.outer_ptrs_);
    inner_indices_ = std::move(other.inner_indices_);

    this->mat_ = other.mat_;
  }
  return *this;
}

template <typename Scalar, sparse_format Compression>
mkl<Scalar, Compression>::~mkl() {
  if (mat_desc_) {
    MATHPRIM_INTERNAL_ASSERT_MKL_SPBLAS(  //
        mkl_sparse_destroy(mat_desc_),   //
        "mkl::~mkl failed");
  }
}

template <typename Scalar, sparse_format Compression>
void mkl<Scalar, Compression>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) {
  MATHPRIM_INTERNAL_CHECK_THROW(mat_desc_, std::runtime_error, "Matrix descriptor is not initialized.");
  if constexpr (std::is_same_v<Scalar, float>) {
    MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(  //
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_desc_, descr_, x.data(), beta, y.data()),
        "mkl::gemv_no_trans failed");
  } else if constexpr (std::is_same_v<Scalar, double>) {
    MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(  //
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat_desc_, descr_, x.data(), beta, y.data()),
        "mkl::gemv_no_trans failed");
  }
}

template <typename Scalar, sparse_format Compression>
void mkl<Scalar, Compression>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) {
  MATHPRIM_INTERNAL_CHECK_THROW(mat_desc_, std::runtime_error, "Matrix descriptor is not initialized.");
  if constexpr (std::is_same_v<Scalar, float>) {
    MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(  //
        mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, alpha, mat_desc_, descr_, x.data(), beta, y.data()),
        "mkl::gemv_trans failed");
  } else if constexpr (std::is_same_v<Scalar, double>) {
    MATHPRIM_INTERNAL_CHECK_MKL_SPBLAS(  //
        mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, alpha, mat_desc_, descr_, x.data(), beta, y.data()),
        "mkl::gemv_trans failed");
  }
}

}  // namespace blas
}  // namespace mathprim::sparse