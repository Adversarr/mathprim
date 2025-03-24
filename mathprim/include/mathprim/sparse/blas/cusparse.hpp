#pragma once
#include <cusparse.h>
#include <iostream>
#include "mathprim/core/defines.hpp"
#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/utils/singleton.hpp"
#include "mathprim/sparse/basic_sparse.hpp"

#define MATHPRIM_CHECK_CUSPARSE(call)                                       \
  do {                                                                      \
    cusparseStatus_t status = call;                                         \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
      printf("cuSPARSE Error at %s:%d - %d\n", __FILE__, __LINE__, status); \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

#if MATHPRIM_OPTION_EXIT_ON_THROW == -1
#  define MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cond, error_type, msg)                                              \
    do {                                                                                                             \
      cusparseStatus_t status = cond;                                                                                \
      if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
        fprintf(stderr, "CUSPARSE Failed: (" #cond ") got %s, at %s:%d\n", cusparseGetErrorString(status), __FILE__, \
                __LINE__);                                                                                           \
        exit(EXIT_FAILURE);                                                                                          \
      }                                                                                                              \
    } while (0)
#elif MATHPRIM_OPTION_EXIT_ON_THROW == 0
#  define MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cond, error_type, msg)                                              \
    do {                                                                                                             \
      cusparseStatus_t status = cond;                                                                                \
      if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
        fprintf(stderr, "CUSPARSE Failed: (" #cond ") got %s, at %s:%d\n", cusparseGetErrorString(status), __FILE__, \
                __LINE__);                                                                                           \
        throw error_type(msg);                                                                                       \
      }                                                                                                              \
    } while (0)
#else
#  define MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cond, error_type, msg)         \
    do {                                                                        \
      cusparseStatus_t status = cond;                                           \
      if (status != CUSPARSE_STATUS_SUCCESS) {                                  \
        throw error_type(msg + std::to_string(cusparseGetErrorString(status))); \
      }                                                                         \
    } while (0)
#endif

namespace mathprim::singletons {
class cusparse_context : public internal::basic_singleton<cusparse_context, cusparseHandle_t> {
  using base = internal::basic_singleton<cusparse_context, cusparseHandle_t>;
  friend base;
  void create_impl(cusparseHandle_t& handle) noexcept {
    MATHPRIM_CHECK_CUSPARSE(cusparseCreate(&handle));

    constexpr int version_compiled = CUSPARSE_VERSION;
    int version_dynamic = 0;
    MATHPRIM_CHECK_CUSPARSE(cusparseGetVersion(handle, &version_dynamic));
    if (version_compiled != version_dynamic) {
      std::cerr << "================================================================================"  //
                << "cuSPARSE version mismatch:\n"                                                      //
                << "  - compiled=" << version_compiled << std::endl                                    //
                << "  - dynamic=" << version_dynamic << std::endl                                      //
                << "Consider static linking, or upgrade your dynamic library to match the compiled\n"  //
                   "version.\n"                                                                        //
                << "================================================================================\n";
      std::exit(EXIT_FAILURE);
    }
  }
  void destroy_impl(cusparseHandle_t& handle) noexcept {
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroy(handle));
  }
};
}  // namespace mathprim::singletons

namespace mathprim::sparse {
namespace blas {

namespace internal {

inline cusparseHandle_t get_cusparse_handle() {
  return singletons::cusparse_context::get();
}

inline bool ensure_dn_mat(cusparseDnMatDescr_t& desc, index_t m, index_t n, index_t ld, void* values,
                          cudaDataType_t data_type, bool row_major = true) {
  bool has_change = false;
  cusparseOrder_t order = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  if (!desc) {
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnMat(&desc, m, n, ld, values, data_type, order));
    has_change = true;
  } else {
    // previous values
    int64_t rows, cols, lda;
    cudaDataType_t type;
    cusparseOrder_t prev_order;
    void* prev_values;
    MATHPRIM_CHECK_CUSPARSE(cusparseDnMatGet(desc, &rows, &cols, &lda, &prev_values, &type, &prev_order));
    if (rows != static_cast<index_t>(m) || cols != static_cast<index_t>(n) || lda != ld || type != data_type
        || order != prev_order) {
      MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnMat(desc));
      MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnMat(&desc, m, n, ld, values, data_type, order));
      has_change = true;
    }
    MATHPRIM_CHECK_CUSPARSE(cusparseDnMatSetValues(desc, values));
  }
  return has_change;  ///< SpMM_preprocess is needed if true.
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
  cusparse() = default;
  explicit cusparse(const_sparse_view mat);
  cusparse(const cusparse&) = delete;
  cusparse& operator=(const cusparse&) = delete;
  cusparse(cusparse&& other);
  cusparse& operator=(cusparse&&);
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
  void reset();

  template <typename SshapeB, typename SstrideB, typename SshapeC, typename SstrideC>
  void spmm_impl(Scalar alpha, basic_view<const Scalar, SshapeB, SstrideB, device::cuda> B, Scalar beta,
                 basic_view<Scalar, SshapeC, SstrideC, device::cuda> C, bool transA);

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

  void prepare_spmm(index_t m, index_t n, index_t k, bool transA, bool transB, bool transC);

  // spmv descriptors
  cusparseSpMatDescr_t mat_desc_{nullptr};
  cusparseDnVecDescr_t x_desc_{nullptr};
  cusparseDnVecDescr_t y_desc_{nullptr};

  // spmm descriptors
  cusparseDnMatDescr_t b_desc_{nullptr};
  cusparseDnMatDescr_t c_desc_{nullptr};

  using temp_buffer = contiguous_vector_buffer<char, device::cuda>;
  std::unique_ptr<temp_buffer> no_transpose_buffer_{nullptr};
  std::unique_ptr<temp_buffer> transpose_buffer_{nullptr};
  std::unique_ptr<temp_buffer> spmm_buffer_{nullptr};
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
  reset();
}

template <typename Scalar, sparse_format Compression>
cusparse<Scalar, Compression>::cusparse(cusparse&& other) : base(other.matrix()) {
  no_transpose_buffer_ = std::move(other.no_transpose_buffer_);
  transpose_buffer_ = std::move(other.transpose_buffer_);
  spmm_buffer_ = std::move(other.spmm_buffer_);
  mat_desc_ = other.mat_desc_;
  x_desc_ = other.x_desc_;
  y_desc_ = other.y_desc_;
  b_desc_ = other.b_desc_;
  c_desc_ = other.c_desc_;
  other.mat_desc_ = nullptr;
  other.x_desc_ = nullptr;
  other.y_desc_ = nullptr;
  other.b_desc_ = nullptr;
  other.c_desc_ = nullptr;
}

template <typename Scalar, sparse_format Compression>
cusparse<Scalar, Compression>& cusparse<Scalar, Compression>::operator=(cusparse&& other) {
  base::operator=(other);
  if (this != &other) {
    reset();
    no_transpose_buffer_ = std::move(other.no_transpose_buffer_);
    transpose_buffer_ = std::move(other.transpose_buffer_);
    spmm_buffer_ = std::move(other.spmm_buffer_);
    mat_desc_ = other.mat_desc_;
    x_desc_ = other.x_desc_;
    y_desc_ = other.y_desc_;
    b_desc_ = other.b_desc_;
    c_desc_ = other.c_desc_;
    other.mat_desc_ = nullptr;
    other.x_desc_ = nullptr;
    other.y_desc_ = nullptr;
    other.b_desc_ = nullptr;
    other.c_desc_ = nullptr;
  }
  return *this;
}

template <typename Scalar, sparse_format Compression>
void cusparse<Scalar, Compression>::reset() {
  if (mat_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroySpMat(mat_desc_));
  if (x_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnVec(x_desc_));
  if (y_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnVec(y_desc_));
  if (b_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnMat(b_desc_));
  if (c_desc_)
    MATHPRIM_CHECK_CUSPARSE(cusparseDestroyDnMat(c_desc_));
  mat_desc_ = nullptr;
  x_desc_ = nullptr;
  y_desc_ = nullptr;
  b_desc_ = nullptr;
  c_desc_ = nullptr;
  no_transpose_buffer_.reset();
  transpose_buffer_.reset();
  spmm_buffer_.reset();
}

template <typename Scalar, sparse_format Compression>
void cusparse<Scalar, Compression>::gemv_no_trans(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) {
  MATHPRIM_INTERNAL_CHECK_THROW(mat_desc_, std::runtime_error, "Matrix descriptor is not initialized.");
  MATHPRIM_ASSERT(x_desc_ && "Vector descriptor is not initialized.");
  MATHPRIM_ASSERT(y_desc_ && "Vector descriptor is not initialized.");
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
  MATHPRIM_INTERNAL_CHECK_THROW(mat_desc_, std::runtime_error, "Matrix descriptor is not initialized.");
  MATHPRIM_ASSERT(x_desc_ && "Vector descriptor is not initialized.");
  MATHPRIM_ASSERT(y_desc_ && "Vector descriptor is not initialized.");

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

template <typename Scalar, sparse_format Compression>
template <typename SshapeB, typename SstrideB, typename SshapeC, typename SstrideC>
void cusparse<Scalar, Compression>::spmm_impl(Scalar alpha, basic_view<const Scalar, SshapeB, SstrideB, device::cuda> B,
                                              Scalar beta, basic_view<Scalar, SshapeC, SstrideC, device::cuda> C,
                                              bool transA) {
  MATHPRIM_INTERNAL_CHECK_THROW(mat_desc_, std::runtime_error, "Matrix descriptor is not initialized.");
  MATHPRIM_ASSERT(x_desc_ && "Vector descriptor is not initialized.");
  MATHPRIM_ASSERT(y_desc_ && "Vector descriptor is not initialized.");
  // Set up the cuSPARSE handle
  cusparseHandle_t handle = internal::get_cusparse_handle();
  const bool trans_b = B.stride(1) != 1, trans_c = C.stride(1) != 1;

  const index_t row_b = trans_b ? B.shape(1) : B.shape(0);
  const index_t col_b = trans_b ? B.shape(0) : B.shape(1);
  const index_t row_c = trans_c ? C.shape(1) : C.shape(0);
  const index_t col_c = trans_c ? C.shape(0) : C.shape(1);
  const index_t ldb = trans_b ? B.stride(1) : B.stride(0);
  const index_t ldc = trans_c ? C.stride(1) : C.stride(0);
  auto data_b = B.data();
  auto data_c = C.data();

  cusparseOperation_t op_a = transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t op_b = trans_b ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  const void* p_alpha = &alpha;
  const void* p_beta = &beta;
  const cudaDataType compute_type = data_type();

  bool has_change = internal::ensure_dn_mat(b_desc_, row_b, col_b, ldb, const_cast<Scalar*>(data_b), compute_type);
  if (trans_c) {  // TODO: check.
    has_change |= internal::ensure_dn_mat(c_desc_, col_c, row_c, ldc, const_cast<Scalar*>(data_c), compute_type, false);
  } else {
    has_change |= internal::ensure_dn_mat(c_desc_, row_c, col_c, ldc, const_cast<Scalar*>(data_c), compute_type);
  }

  if (has_change) {
    size_t buffer_size = 0;
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, op_a, op_b, &alpha, mat_desc_, b_desc_, &beta, c_desc_,
                                                    compute_type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
    spmm_buffer_ = std::make_unique<temp_buffer>(make_cuda_buffer<char>(buffer_size));
    void* ext_buffer = spmm_buffer_->data();
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMM_preprocess(handle, op_a, op_b, p_alpha, mat_desc_, b_desc_, p_beta, c_desc_,
                                                    compute_type, CUSPARSE_SPMM_ALG_DEFAULT, ext_buffer));
  }

  MATHPRIM_CHECK_CUSPARSE(cusparseSpMM(handle, op_a, op_b, p_alpha, mat_desc_, b_desc_, p_beta, c_desc_, compute_type,
                                       CUSPARSE_SPMM_ALG_DEFAULT, spmm_buffer_->data()));
}

}  // namespace blas
}  // namespace mathprim::sparse
