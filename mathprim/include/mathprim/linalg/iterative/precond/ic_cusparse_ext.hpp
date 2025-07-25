#pragma once
#include <cusparse.h>

#include <stdexcept>

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"
#include "mathprim/sparse/basic_sparse.hpp"
#include "mathprim/sparse/blas/cusparse.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace mathprim::sparse::iterative {

template <typename Scalar, typename Device, sparse::sparse_format SparseCompression>
class cusparse_ichol_ext;

template <typename Scalar>
class cusparse_ichol_ext<Scalar, device::cuda, sparse::sparse_format::csr>
    : public basic_preconditioner<cusparse_ichol_ext<Scalar, device::cuda, sparse::sparse_format::csr>, Scalar,
                                  device::cuda, sparse_format::csr> {
public:
  using base = basic_preconditioner<cusparse_ichol_ext<Scalar, device::cuda, sparse::sparse_format::csr>, Scalar,
                                    device::cuda, sparse_format::csr>;
  friend base;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse = sparse::basic_sparse_view<const Scalar, device::cuda, sparse::sparse_format::csr>;
  static constexpr bool is_float32 = std::is_same_v<Scalar, float>;
  static_assert(is_float32 || std::is_same_v<Scalar, double>, "Only float32 and float64 are supported.");

  cusparse_ichol_ext() = default;
  explicit cusparse_ichol_ext(const_sparse view) : base(view) { this->compute({}); }

  cusparse_ichol_ext(cusparse_ichol_ext&& other) :
      base(other),
      descr_a_(other.descr_a_),
      descr_sparse_a_(other.descr_sparse_a_),
      descr_sparse_lower_(other.descr_sparse_lower_),
      buffer_intern_(std::move(other.buffer_intern_)),
      buffer_l_(std::move(other.buffer_l_)),
      buffer_lt_(std::move(other.buffer_lt_)),
      buffer_chol_(std::move(other.buffer_chol_)),
      spsvDescrL_(other.spsvDescrL_),
      spsvDescrLtrans_(other.spsvDescrLtrans_),
      vec_x_(other.vec_x_),
      vec_y_(other.vec_y_),
      vec_intern_(other.vec_intern_) {
    other.descr_a_ = nullptr;
    other.descr_sparse_a_ = nullptr;
    other.descr_sparse_lower_ = nullptr;
    other.spsvDescrL_ = nullptr;
    other.spsvDescrLtrans_ = nullptr;
    other.vec_x_ = nullptr;
    other.vec_y_ = nullptr;
    other.vec_intern_ = nullptr;
  }

  ~cusparse_ichol_ext() { reset(); }

  void analyze_impl() {
    reset();  // Clear all previous information

    if (!static_cast<bool>(lower_)) {
      // external is not set.
      return;
    }

    cusparseHandle_t handle = sparse::blas::internal::get_cusparse_handle();
    auto matrix = this->matrix();
    index_t rows = matrix.rows();
    index_t nnz = matrix.nnz();
    cudaDataType_t dtype = std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;

    /* Description of the A matrix */
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cusparseCreateMatDescr(&descr_a_), std::runtime_error,
                                           "Failed to create matrix descriptor.");
    cusparseMatrixType_t mat_type = CUSPARSE_MATRIX_TYPE_GENERAL;

    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cusparseSetMatType(descr_a_, mat_type), std::runtime_error,
                                           "Failed to set matrix type.");
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cusparseSetMatIndexBase(descr_a_, CUSPARSE_INDEX_BASE_ZERO),
                                           std::runtime_error, "Failed to set matrix index base.");

    /* Allocate required memory */
    /* Wrap raw data into cuSPARSE generic API objects */
    buffer_intern_ = make_cuda_buffer<Scalar>(rows);
    buffer_x_ = make_cuda_buffer<Scalar>(rows);
    buffer_y_ = make_cuda_buffer<Scalar>(rows);
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&vec_intern_, rows, buffer_intern_.data(), dtype));
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x_, rows, buffer_x_.data(), dtype));
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y_, rows, buffer_y_.data(), dtype));

    /* Initialize problem data */
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    // Lower Part
    {
      auto row_offsets = lower_.outer_ptrs().data();
      auto col_indices = lower_.inner_indices().data();
      auto values = lower_.values().data();
      // directly set from the external matrix
      MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(
          cusparseCreateCsr(                                                     //
              &descr_sparse_lower_,                                              // descr
              rows, rows, nnz,                                                   // shape
              row_offsets, col_indices, values,                                  // content
              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,  // indexing
              dtype),
          std::runtime_error, "Failed to create triangular matrix in Cholesky");
    }
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_sparse_lower_,  //
                                                      CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower)));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_sparse_lower_,  //
                                                      CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit)));


    /* Allocate workspace for cuSPARSE */
    size_t buf_size;
    const Scalar floatone = 1;

    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                                    descr_sparse_lower_, vec_x_, vec_intern_, dtype,
                                                    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL_, &buf_size));
    buffer_l_ = make_cuda_buffer<char>(buf_size);

    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrLtrans_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, &floatone,
                                                    descr_sparse_lower_, vec_intern_, vec_y_, dtype,
                                                    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLtrans_, &buf_size));
    buffer_lt_ = make_cuda_buffer<char>(buf_size);
  }

  void factorize_impl() {
    if (!static_cast<bool>(lower_)) {
      // external is not set.
      return;
    }

    cusparseHandle_t handle = sparse::blas::internal::get_cusparse_handle();
    constexpr cudaDataType_t dtype = std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;
    const Scalar floatone = 1;

    /* Perform triangular solve analysis */
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_analysis(            //
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,  // solve info
        descr_sparse_lower_, vec_x_, vec_intern_, dtype,      // matrix info
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL_, buffer_l_.data()));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_analysis(        //
        handle, CUSPARSE_OPERATION_TRANSPOSE, &floatone,  // solve info
        descr_sparse_lower_, vec_intern_, vec_y_, dtype,  // matrix info
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLtrans_, buffer_lt_.data()));
  }

  /**
   * @brief Set the preconditioner from an external lower triangular matrix as IC factorization.
   *
   * @tparam OtherDevice
   * @param other
   */
  template <typename OtherDevice>
  void set_external(
    const basic_sparse_view<const Scalar, OtherDevice, sparse::sparse_format::csr>& other
  ) {
    // Reset any existing data
    reset();
    MATHPRIM_ASSERT(other.rows() == other.cols() && "The matrix must be square.");

    // Store a copy of the external matrix data on CUDA device
    index_t rows = other.rows();
    index_t nnz = other.nnz();
    
    auto ext_row_offsets = make_cuda_buffer<index_t>(rows + 1);
    auto ext_col_indices = make_cuda_buffer<index_t>(nnz);
    auto ext_values = make_cuda_buffer<Scalar>(nnz);
    
    // Copy data from the input matrix to the device buffers
    copy(ext_values.view(), other.values());
    copy(ext_row_offsets.view(), other.outer_ptrs());
    copy(ext_col_indices.view(), other.inner_indices());

    lower_ = sparse::basic_sparse_matrix<Scalar, device::cuda, mathprim::sparse::sparse_format::csr>(
        std::move(ext_values),       // values
        std::move(ext_row_offsets),  // outer_ptrs
        std::move(ext_col_indices),  // inner_indices
        rows,                        // rows
        rows,                        // cols, equal.
        nnz,                         // nnz
        sparse_property::general     // property
    );

    // Run analysis and factorization
    analyze_impl();
    factorize_impl();
  }

  void reset() noexcept {
    if (descr_a_) {
      cusparseDestroyMatDescr(descr_a_);
      descr_a_ = nullptr;
    }
    if (descr_sparse_a_) {
      cusparseDestroySpMat(descr_sparse_a_);
      descr_sparse_a_ = nullptr;
    }
    if (descr_sparse_lower_) {
      cusparseDestroySpMat(descr_sparse_lower_);
      descr_sparse_lower_ = nullptr;
    }

    if (vec_x_) {
      cusparseDestroyDnVec(vec_x_);
      vec_x_ = nullptr;
    }
    if (vec_y_) {
      cusparseDestroyDnVec(vec_y_);
      vec_y_ = nullptr;
    }
    if (vec_intern_) {
      cusparseDestroyDnVec(vec_intern_);
      vec_intern_ = nullptr;
    }
    if (spsvDescrL_) {
      cusparseSpSV_destroyDescr(spsvDescrL_);
      spsvDescrL_ = nullptr;
    }
    if (spsvDescrLtrans_) {
      cusparseSpSV_destroyDescr(spsvDescrLtrans_);
      spsvDescrLtrans_ = nullptr;
    }
  }

private:
  void apply_impl(vector_type y, const_vector x) {
    cusparseHandle_t handle = sparse::blas::internal::get_cusparse_handle();
    Scalar floatone = 1;
    auto dtype = std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;
    copy(buffer_x_.view(), x);
    auto no_trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto trans = CUSPARSE_OPERATION_TRANSPOSE;
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_solve(           //
        handle,                                           // context
        no_trans, &floatone,                              // info
        descr_sparse_lower_, vec_x_, vec_intern_, dtype,  // descriptors
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_solve(           //
        handle,                                           // context
        trans, &floatone,                                 // info
        descr_sparse_lower_, vec_intern_, vec_y_, dtype,  // descriptors
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLtrans_));
    copy(y, buffer_y_.view());
  }

  // Matrix A:
  cusparseMatDescr_t descr_a_{nullptr};
  cusparseSpMatDescr_t descr_sparse_a_{nullptr};
  // Matrix L: A = L U
  cusparseSpMatDescr_t descr_sparse_lower_{nullptr};

  contiguous_buffer<Scalar, dshape<1>, device::cuda> buffer_intern_;
  contiguous_buffer<char, dshape<1>, device::cuda> buffer_l_, buffer_lt_, buffer_chol_;

  // External matrix data storage
  basic_sparse_matrix<Scalar, device::cuda, sparse::sparse_format::csr> lower_;

  contiguous_vector_buffer<Scalar, device::cuda> buffer_x_;
  contiguous_vector_buffer<Scalar, device::cuda> buffer_y_;

  // solver
  cusparseSpSVDescr_t spsvDescrL_{nullptr}, spsvDescrLtrans_{nullptr};

  // inputs
  cusparseDnVecDescr_t vec_x_{nullptr}, vec_y_{nullptr}, vec_intern_{nullptr};
};

}  // namespace mathprim::sparse::iterative

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
