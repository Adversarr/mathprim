#pragma once
#include <cusparse.h>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
namespace mathprim::sparse {
namespace blas {

// TODO: replace exit with exception.
#define MATHPRIM_CHECK_CUSPARSE(call)                                       \
  do {                                                                      \
    cusparseStatus_t status = call;                                         \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
      printf("cuSPARSE Error at %s:%d - %d\n", __FILE__, __LINE__, status); \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

namespace internal {
class cusparse_handle {
public:
  static cusparse_handle& instance() {
    static cusparse_handle handle;
    return handle;
  }

  cusparseHandle_t get() const {
    return handle_;
  }

private:
  cusparseHandle_t handle_;
  cusparse_handle() {
    MATHPRIM_CHECK_CUSPARSE(cusparseCreate(&handle_));
  }

  ~cusparse_handle() {
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroy(handle_));
  }
};

inline cusparseHandle_t get_cusparse_handle() {
  return cusparse_handle::instance().get();
}

}  // namespace internal

template <typename Scalar, sparse_format Compression>
class cusparse : public sparse_blas_base<cusparse<Scalar, Compression>, Scalar, device::cuda, Compression> {
public:
  using base = sparse_blas_base<cusparse<Scalar, Compression>, Scalar, device::cuda, Compression>;
  friend base;
  using vector_view = typename base::vector_view;
  using const_vector_view = typename base::const_vector_view;
  using sparse_view = typename base::sparse_view;
  using const_sparse_view = typename base::const_sparse_view;
  explicit cusparse(const_sparse_view mat);
  cusparse(const cusparse&) = delete;
  cusparse(cusparse&& other);
  cusparse& operator=(const cusparse&) = delete;
  cusparse& operator=(cusparse&&) = delete;
  ~cusparse();

  static constexpr cusparseIndexType_t index_type() {
#if MATHPRIM_USE_LONG_INDEX
    return CUSPARSE_INDEX_64I;
#else
    return CUSPARSE_INDEX_32I;
#endif
  }

  static constexpr cudaDataType_t data_type() {
    if constexpr (std::is_same_v<Scalar, float>) {
      return CUDA_R_32F;
    } else if constexpr (std::is_same_v<Scalar, double>) {
      return CUDA_R_64F;
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Unsupported scalar type");
    }
  }

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

  cusparseSpMatDescr_t mat_desc_{nullptr};
  cusparseDnVecDescr_t x_desc_{nullptr};
  cusparseDnVecDescr_t y_desc_{nullptr};

  using temp_buffer = continuous_buffer<char, shape_t<keep_dim>, device::cuda>;
  std::unique_ptr<temp_buffer> no_transpose_buffer_;
  std::unique_ptr<temp_buffer> transpose_buffer_;
};

///////////////////////////////////////////////////////////////////////////////
/// Implementation for CSR format.
///////////////////////////////////////////////////////////////////////////////

template <typename Scalar, sparse_format Compression>
cusparse<Scalar, Compression>::cusparse(const_sparse_view mat) : base(mat) {
  int64_t rows = this->mat_.rows();
  int64_t cols = this->mat_.cols();
  int64_t nnz = this->mat_.nnz();
  void* values = const_cast<Scalar*>(this->mat_.values().data());
  void* inner = const_cast<index_t*>(this->mat_.inner_indices().data());
  void* outer = const_cast<index_t*>(this->mat_.outer_ptrs().data());
  auto data_type = this->data_type();
  if constexpr (Compression == sparse_format::csr) {
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateCsr(&mat_desc_, rows, cols, nnz, outer, inner, values, index_type(),
                                              index_type(), CUSPARSE_INDEX_BASE_ZERO, data_type));
  } else {
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateCsc(&mat_desc_, rows, cols, nnz, outer, inner, values, index_type(),
                                              index_type(), CUSPARSE_INDEX_BASE_ZERO, data_type));
  }
  MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&x_desc_, cols, nullptr, data_type));
  MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&y_desc_, rows, nullptr, data_type));
}

template <typename Scalar, sparse_format Compression>
cusparse<Scalar, Compression>::~cusparse() {
  if (mat_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroySpMat(mat_desc_));
  if (x_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnVec(x_desc_));
  if (y_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnVec(y_desc_));
}

template <typename Scalar, sparse_format Compression>
cusparse<Scalar, Compression>::cusparse(cusparse&& other) : base(other.matrix()) {
  mat_desc_ = other.mat_desc_;
  x_desc_ = other.x_desc_;
  y_desc_ = other.y_desc_;
  no_transpose_buffer_ = std::move(other.no_transpose_buffer_);
  transpose_buffer_ = std::move(other.transpose_buffer_);
  other.mat_desc_ = nullptr;
  other.x_desc_ = nullptr;
  other.y_desc_ = nullptr;
}

template <typename Scalar, sparse_format Compression>
void cusparse<Scalar, Compression>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) {
  // Set up the cuSPARSE handle
  cusparseHandle_t handle = internal::get_cusparse_handle();

  // Set up the operation descriptor
  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // Set up the cuSPARSE SpMV descriptor
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

  // Set up the cuSPARSE SpMV descriptor
  cusparseSpMatDescr_t mat_desc = mat_desc_;
  cusparseDnVecDescr_t x_desc = x_desc_;
  cusparseDnVecDescr_t y_desc = y_desc_;
  // Set the pointers for x and y
  MATHPRIM_CHECK_CUSPARSE(cusparseDnVecSetValues(x_desc, const_cast<Scalar*>(x.data())));
  MATHPRIM_CHECK_CUSPARSE(cusparseDnVecSetValues(y_desc, const_cast<Scalar*>(y.data())));

  // External buffer for the SpMV operation
  if (!no_transpose_buffer_) {
    size_t buffer_size = 0;
    MATHPRIM_CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(handle, op, &alpha, mat_desc, x_desc, &beta, y_desc, data_type(), alg, &buffer_size));
    no_transpose_buffer_ = std::make_unique<temp_buffer>(make_buffer<char, device::cuda>(buffer_size));
  }

  // Perform the SpMV operation
  MATHPRIM_CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, mat_desc, x_desc, &beta, y_desc, data_type(), alg,
                                       no_transpose_buffer_->data()));
}

template <typename Scalar, sparse_format Compression>
void cusparse<Scalar, Compression>::gemv_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) {
  // Set up the cuSPARSE handle
  cusparseHandle_t handle = internal::get_cusparse_handle();

  // Set up the operation descriptor
  cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;

  // Set up the cuSPARSE SpMV descriptor
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

  // Set up the cuSPARSE SpMV descriptor
  cusparseSpMatDescr_t mat_desc = mat_desc_;
  cusparseDnVecDescr_t x_desc = x_desc_;
  cusparseDnVecDescr_t y_desc = y_desc_;
  // Set the pointers for x and y
  MATHPRIM_CHECK_CUSPARSE(cusparseDnVecSetValues(x_desc, const_cast<Scalar*>(x.data())));
  MATHPRIM_CHECK_CUSPARSE(cusparseDnVecSetValues(y_desc, const_cast<Scalar*>(y.data())));
  // External buffer for the SpMV operation
  if (!transpose_buffer_) {
    size_t buffer_size = 0;
    MATHPRIM_CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(handle, op, &alpha, mat_desc, x_desc, &beta, y_desc, data_type(), alg, &buffer_size));
    transpose_buffer_ = std::make_unique<temp_buffer>(make_buffer<char, device::cuda>(buffer_size));
  }

  // Perform the SpMV operation
  MATHPRIM_CHECK_CUSPARSE(
      cusparseSpMV(handle, op, &alpha, mat_desc, x_desc, &beta, y_desc, data_type(), alg, transpose_buffer_->data()));
}

}  // namespace blas
}  // namespace mathprim::sparse