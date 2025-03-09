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
class ilu;

template <typename Scalar>
class ilu<Scalar, device::cuda, sparse::sparse_format::csr>
    : public basic_preconditioner<ilu<Scalar, device::cuda, sparse::sparse_format::csr>, Scalar, device::cuda> {

public:
  using base = basic_preconditioner<ilu<Scalar, device::cuda, sparse::sparse_format::csr>, Scalar, device::cuda>;
  friend base;
  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  using const_sparse_view = sparse::basic_sparse_view<const Scalar, device::cuda, sparse::sparse_format::csr>;
  static constexpr bool is_float32 = std::is_same_v<Scalar, float>;
  static_assert(is_float32 || std::is_same_v<Scalar, double>, "Only float32 and float64 are supported.");

  ilu();
  ilu(const const_sparse_view& view, bool need_compute = true) :  // NOLINT(google-explicit-constructor)
      matrix_(view) {
    if (need_compute) {
      compute();
    }
  }

  ilu(ilu&& other): 
    matrix_(other.matrix_),
    descr_a_(other.descr_a_),
    descr_sparse_a_(other.descr_sparse_a_),
    descr_sparse_lower_(other.descr_sparse_lower_), 
    descr_sparse_upper_(other.descr_sparse_upper_),
    descr_lu_(other.descr_lu_),
    info_ilu_(other.info_ilu_),
    ilu_nnz_copy_(std::move(other.ilu_nnz_copy_)),
    buffer_intern_(std::move(other.buffer_intern_)),
    buffer_l_(std::move(other.buffer_l_)),
    buffer_u_(std::move(other.buffer_u_)),
    buffer_lu_(std::move(other.buffer_lu_)),
    spsvDescrL_(other.spsvDescrL_),
    spsvDescrU_(other.spsvDescrU_),
    vec_x_(other.vec_x_), vec_y_(other.vec_y_), vec_intern_(other.vec_intern_) {
    other.descr_a_ = nullptr;
    other.descr_sparse_a_ = nullptr;
    other.descr_sparse_lower_ = nullptr;
    other.descr_sparse_upper_ = nullptr;
    other.descr_lu_ = nullptr;
    other.info_ilu_ = nullptr;
    other.spsvDescrL_ = nullptr;
    other.spsvDescrU_ = nullptr;
    other.vec_x_ = nullptr;
    other.vec_y_ = nullptr;
    other.vec_intern_ = nullptr;
  }

  ~ilu() {
    reset();
  }

  void analyze_pattern() {
    reset(); // clear all previous information
    cusparseHandle_t handle = sparse::blas::internal::get_cusparse_handle();
    index_t rows = matrix_.rows();
    index_t nnz = matrix_.nnz();
    cudaDataType_t dtype = std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;

    /* Description of the A matrix */
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(
      cusparseCreateMatDescr(&descr_a_),
      std::runtime_error, "Failed to create matrix descriptor.");
    cusparseMatrixType_t mat_type = CUSPARSE_MATRIX_TYPE_GENERAL;
    // if (matrix_.property() == sparse::sparse_property::symmetric) {
    //   mat_type = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
    // }
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(
      cusparseSetMatType(descr_a_, mat_type),
      std::runtime_error, "Failed to set matrix type.");
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(
      cusparseSetMatIndexBase(descr_a_, CUSPARSE_INDEX_BASE_ZERO),
      std::runtime_error, "Failed to set matrix index base.");

    /* Allocate required memory */
    ilu_nnz_copy_ = make_cuda_buffer<float>(nnz);
    /* Wrap raw data into cuSPARSE generic API objects */
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x_, rows, nullptr, dtype));
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y_, rows, nullptr, dtype));

    /* Initialize problem data */
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    
    index_t* row_offsets = const_cast<index_t*>(matrix_.outer_ptrs().data());
    index_t* col_indices = const_cast<index_t*>(matrix_.inner_indices().data());
    Scalar* values = const_cast<Scalar*>(matrix_.values().data());
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateCsr(                             //
        &descr_sparse_a_,                                                  // descr
        rows, rows, nnz,                                                   // shape
        row_offsets, col_indices, values,                                  // content
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,  // indexing
        dtype));

    /* Copy A data to ILU(0) vals as input*/
    copy(ilu_nnz_copy_.view(), matrix_.values().as_const());
    // Lower Part
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cusparseCreateCsr(                                                     //
        &descr_sparse_lower_,                                              // descr
        rows, rows, nnz,                                                   // shape
        row_offsets, col_indices, ilu_nnz_copy_.data(),                    // content
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,  // indexing
        dtype), std::runtime_error, "Failed to create triangular matrix in ILU");
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_sparse_lower_,  //
                                                      CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower)));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_sparse_lower_,  //
                                                      CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit)));
    // Upper Part
    MATHPRIM_INTERNAL_CHECK_THROW_CUSPARSE(cusparseCreateCsr(                                                     //
        &descr_sparse_upper_,                                              // descr
        rows, rows, nnz,                                                   // shape
        row_offsets, col_indices, ilu_nnz_copy_.data(),                    // content
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,  // indexing
        dtype), std::runtime_error, "Failed to create triangular matrix in ILU");
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_sparse_upper_,  //
                                                      CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper)));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr_sparse_upper_,  //
                                                      CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit)));

    /* Create ILU(0) info object */
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateCsrilu02Info(&info_ilu_));
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_lu_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSetMatType(descr_lu_, CUSPARSE_MATRIX_TYPE_GENERAL));
    MATHPRIM_CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_lu_, CUSPARSE_INDEX_BASE_ZERO));

    /* Allocate workspace for cuSPARSE */
    size_t buf_size_l, buf_size_u;
    float floatone = 1;
    int requirement;
    if constexpr (is_float32) {
      MATHPRIM_CHECK_CUSPARSE(cusparseScsrilu02_bufferSize(handle,                                       // call
                                                           rows, nnz,                                    // shape
                                                           descr_lu_, values, row_offsets, col_indices,  // content
                                                           info_ilu_, &requirement));
    } else {
      MATHPRIM_CHECK_CUSPARSE(cusparseDcsrilu02_bufferSize(handle,                                       // call
                                                           rows, nnz,                                    // shape
                                                           descr_lu_, values, row_offsets, col_indices,  // content
                                                           info_ilu_, &requirement));
    }
    buffer_lu_ = make_cuda_buffer<char>(requirement);
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, descr_sparse_lower_, vec_x_, vec_y_,
                            dtype, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL_, &buf_size_l));

    buffer_l_ = make_cuda_buffer<char>(buf_size_l);
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrU_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, descr_sparse_upper_, vec_x_, vec_y_,
                            dtype, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU_, &buf_size_u));
    buffer_u_ = make_cuda_buffer<char>(buf_size_u);
  }

  void factorize() {
    cusparseHandle_t handle = sparse::blas::internal::get_cusparse_handle();
    index_t rows = matrix_.rows();
    index_t nnz = matrix_.nnz();
    cudaDataType_t dtype = std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;
    index_t* row_offsets = const_cast<index_t*>(matrix_.outer_ptrs().data());
    index_t* col_indices = const_cast<index_t*>(matrix_.inner_indices().data());
    Scalar* values = ilu_nnz_copy_.data();
    float floatone = 1;
    /* Copy A data to Cholesky vals as input*/
    copy(ilu_nnz_copy_.view(), matrix_.values().as_const());
    /* Perform analysis for ILU(0) */
    if constexpr (is_float32) {
      MATHPRIM_CHECK_CUSPARSE(cusparseScsrilu02_analysis(  //
          handle,                                          // context
          rows, nnz, descr_a_,                             // matrix to factorize
          values, row_offsets, col_indices,                // matrix data
          info_ilu_, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer_lu_.data()));
      /* generate the ILU(0) factors */
      MATHPRIM_CHECK_CUSPARSE(cusparseScsrilu02(  //
          handle,                                 // context
          rows, nnz, descr_lu_,                   // matrix
          values, row_offsets, col_indices,       // matrix data
          info_ilu_, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer_lu_.data()));
    } else {
      MATHPRIM_CHECK_CUSPARSE(cusparseDcsrilu02_analysis(  //
          handle,                                          // context
          rows, nnz, descr_a_,                             // matrix to factorize
          values, row_offsets, col_indices,                // matrix data
          info_ilu_, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer_lu_.data()));
      /* generate the ILU(0) factors */
      MATHPRIM_CHECK_CUSPARSE(cusparseDcsrilu02(  //
          handle,                                 // context
          rows, nnz, descr_lu_,                   // matrix
          values, row_offsets, col_indices,       // matrix data
          info_ilu_, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer_lu_.data()));
    }
    /* perform triangular solve analysis */
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_analysis(            //
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,  // solve info
        descr_sparse_lower_, vec_x_, vec_y_, dtype,           // matrix info
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL_, buffer_l_.data()));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_analysis(            //
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,  // solve info
        descr_sparse_upper_, vec_x_, vec_y_, dtype,           // matrix info
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU_, buffer_u_.data()));
    buffer_intern_ = make_cuda_buffer<float>(rows);
    MATHPRIM_CHECK_CUSPARSE(cusparseCreateDnVec(&vec_intern_, rows, buffer_intern_.data(), dtype));
  }

  const const_sparse_view& matrix() {
    return matrix_;
  }

  void set_matrix(const const_sparse_view& view) {
    matrix_ = view;
    reset();
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
    if (descr_sparse_upper_) {
      cusparseDestroySpMat(descr_sparse_upper_);
      descr_sparse_upper_ = nullptr;
    }
    if (descr_lu_) {
      cusparseDestroyMatDescr(descr_lu_);
      descr_lu_ = nullptr;
    }
    if (info_ilu_) {
      MATHPRIM_CHECK_CUSPARSE(cusparseDestroyCsrilu02Info(info_ilu_));
      info_ilu_ = nullptr;
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
    if (spsvDescrU_) {
      cusparseSpSV_destroyDescr(spsvDescrU_);
      spsvDescrU_ = nullptr;
    }
  }

  void compute() {
    analyze_pattern();
    factorize();
  }

private:
  void apply_impl(vector_type y, const_vector x) {
    cusparseHandle_t handle = sparse::blas::internal::get_cusparse_handle();
    float floatone = 1;
    auto dtype = std::is_same_v<Scalar, float> ? CUDA_R_32F : CUDA_R_64F;
    MATHPRIM_CHECK_CUSPARSE(cusparseDnVecSetValues(vec_x_, const_cast<Scalar*>(x.data()))); // input
    MATHPRIM_CHECK_CUSPARSE(cusparseDnVecSetValues(vec_y_, y.data())); // output
    auto no_trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_solve(           //
        handle,                                           // context
        no_trans, &floatone,                              // info
        descr_sparse_lower_, vec_x_, vec_intern_, dtype,  // descriptors
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL_));
    MATHPRIM_CHECK_CUSPARSE(cusparseSpSV_solve(           //
        handle,                                           // context
        no_trans, &floatone,                              // info
        descr_sparse_upper_, vec_intern_, vec_y_, dtype,  // descriptors
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU_));
  }

  // Matrix A:
  const_sparse_view matrix_;
  cusparseMatDescr_t descr_a_{nullptr};
  cusparseSpMatDescr_t descr_sparse_a_{nullptr};
  // Matrix L: A = L U
  cusparseSpMatDescr_t descr_sparse_lower_{nullptr}, descr_sparse_upper_{nullptr};
  cusparseMatDescr_t descr_lu_{nullptr};
  csrilu02Info_t info_ilu_{nullptr};
  cusparseSpMatDescr_t descr_lower_{nullptr};
  cusparseSpMatDescr_t descr_upper_{nullptr};
  contiguous_buffer<Scalar, dshape<1>, device::cuda> ilu_nnz_copy_;
  contiguous_buffer<Scalar, dshape<1>, device::cuda> buffer_intern_;
  contiguous_buffer<char, dshape<1>, device::cuda> buffer_l_, buffer_u_, buffer_lu_;


  // solver
  cusparseSpSVDescr_t spsvDescrL_{nullptr}, spsvDescrU_{nullptr};

  // inputs
  cusparseDnVecDescr_t vec_x_=nullptr, vec_y_ = nullptr, vec_intern_ = nullptr;
};

}  // namespace mathprim::sparse::iterative

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif