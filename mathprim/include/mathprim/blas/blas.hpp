#pragma once
#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
#include "mathprim/core/view.hpp"

namespace mathprim {
namespace blas {
namespace internal {

template <index_t arow, index_t acol, index_t brow, index_t bcol, index_t crow, index_t ccol>
void check_mm_shapes(const shape_t<arow, acol> &a, const shape_t<brow, bcol> &b, const shape_t<crow, ccol> &c) {
auto [an, am] = a;
auto [bn, bm] = b;
auto [cn, cm] = c;

MATHPRIM_INTERNAL_CHECK_THROW(am == bn, shape_error, "blas::gemm: A.shape(1) != B.shape(0)");
MATHPRIM_INTERNAL_CHECK_THROW(an == cn, shape_error, "blas::gemm: A.shape(0) != C.shape(0)");
MATHPRIM_INTERNAL_CHECK_THROW(bm == cm, shape_error, "blas::gemm: B.shape(1) != C.shape(1)");
}

template <index_t arow, index_t acol, index_t xrow, index_t yrow>
void check_mv_shapes(const shape_t<arow, acol> &a, const shape_t<xrow> &x, const shape_t<yrow> &y) {
auto [an, am] = a;
auto [xn] = x;
auto [yn] = y;

MATHPRIM_INTERNAL_CHECK_THROW(am == xn, shape_error, "blas::gemv: A.shape(1) != x.shape(0)");
MATHPRIM_INTERNAL_CHECK_THROW(an == yn, shape_error, "blas::gemv: A.shape(0) != y.shape(0)");
}

template <typename Scalar, index_t... sshape_values, index_t... sstride_values>
constexpr bool is_capable_vector(const shape_t<sshape_values...> &shape,
                                const stride_t<sstride_values...> &stride) noexcept {
if constexpr (sizeof...(sshape_values) == 1) {
  return true;
} else {
  // Only the last stride can vary.
  const index_t last_stride = stride.template get<-1>();
  const index_t last_stride_elem = last_stride;
  const auto default_stride = make_default_stride<Scalar>(shape).to_array();
  return last_stride_elem * default_stride == stride.to_array();
}
}

template <typename Scalar, index_t srows, index_t scols, index_t lda, index_t elem>
constexpr bool is_capable_matrix(const shape_t<srows, scols> &shape, const stride_t<lda, elem> &stride) noexcept {
auto [rows, cols] = shape;
auto [lda_val, elem_val] = stride;
return elem_val == 1 && lda_val >= cols;
}

template <typename Scalar, typename sshape, typename sstride, typename device>
constexpr bool is_capable_vector(basic_view<Scalar, sshape, sstride, device> view) noexcept {
return is_capable_vector<Scalar>(view.shape(), view.stride());
}

enum class matrix_op {
invalid,
none,       // CBLAS_TRANSPOSE::CblasNoTrans
transpose,  // CBLAS_TRANSPOSE::CblasTrans
};

template <typename Scalar, typename sshape, typename sstride, typename device>
constexpr matrix_op get_matrix_op(basic_view<Scalar, sshape, sstride, device> view) noexcept {
bool is_none_ok = is_capable_matrix<Scalar>(view.shape(), view.stride());
auto transpose_view = view.transpose();
bool is_transpose_ok = is_capable_matrix<Scalar>(transpose_view.shape(), transpose_view.stride());

if (is_none_ok) {
  return matrix_op::none;
} else if (is_transpose_ok) {
  return matrix_op::transpose;
} else {
  return matrix_op::invalid;
}
}

}  // namespace internal

template <typename Derived, typename Scalar, typename Dev>
struct basic_blas {
using scalar_type = Scalar;
using device_type = Dev;
using impl_type = Derived;

template <typename T, typename Sshape, typename Sstride>
using generic_type = mathprim::basic_view<T, Sshape, Sstride, Dev>;

template <typename Sshape, typename Sstride>
using view_type = basic_view<Scalar, Sshape, Sstride, Dev>;
template <typename Sshape, typename Sstride>
using const_type = basic_view<const Scalar, Sshape, Sstride, Dev>;

template <typename T>
static constexpr bool is_same_scalar_v = std::is_same_v<std::remove_const_t<T>, Scalar>;

template <typename ...T>
using enable_if_same_scalar_t = std::enable_if_t<(is_same_scalar_v<T> && ...)>;

/**
  * @brief Copy the elements of src to dst.
  *
  * @param dst
  * @param src
  */
template </* ScalarDst == Scalar */ typename SshapeDst, typename SstrideDst,  // dst
          typename ScalarSrc, typename SshapeSrc, typename SstrideSrc,        // src
          typename = enable_if_same_scalar_t<ScalarSrc>>
MATHPRIM_NOINLINE void copy(view_type<SshapeDst, SstrideDst> dst,
                            generic_type<ScalarSrc, SshapeSrc, SstrideSrc> src) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(dst), std::runtime_error,
                                "Incompatible views for BLAS copy. (dst)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(src), std::runtime_error,
                                "Incompatible views for BLAS copy. (src)");

  // assert shape equal.
  MATHPRIM_INTERNAL_CHECK_THROW(dst.shape() == src.shape(), shape_error, "blas::copy: dst.shape() != src.shape()");

  static_cast<Derived *>(this)->copy_impl(dst.flatten(), src.as_const().flatten());
}

/**
  * @brief Scale the elements of x by alpha.
  *
  * @param alpha
  * @param x
  */
template <typename Sshape, typename Sstride>
MATHPRIM_NOINLINE void scal(Scalar alpha, view_type<Sshape, Sstride> x) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS scal.");

  static_cast<Derived *>(this)->scal_impl(alpha, x.flatten());
}

/**
  * @brief Swap the elements of x and y.
  *
  * @param x
  * @param y
  */
template <typename Sshape, typename Sstride>
MATHPRIM_NOINLINE void swap(view_type<Sshape, Sstride> x, view_type<Sshape, Sstride> y) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS swap. (x)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                "Incompatible views for BLAS swap. (y)");

  // assert shape equal.
  MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::swap: x.shape() != y.shape()");

  static_cast<Derived *>(this)->swap_impl(x.flatten(), y.flatten());
}

/**
  * @brief Compute y <- alpha * x + y.
  *
  * @param alpha
  * @param x
  * @param y
  */
template <typename ScalarX, typename SshapeX, typename SstrideX,  // x
          typename SshapeY, typename SstrideY,                    // y
          typename = enable_if_same_scalar_t<ScalarX>>
MATHPRIM_NOINLINE void axpy(Scalar alpha, generic_type<ScalarX, SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS axpy. (x)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                "Incompatible views for BLAS axpy. (y)");

  MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::axpy: x.shape() != y.shape()");

  static_cast<Derived *>(this)->axpy_impl(alpha, x.as_const().flatten(), y.flatten());
}

/**
  * @brief Compute the dot product of x and y.
  *
  * @param x
  * @param y
  * @return Scalar
  */
template <typename ScalarX, typename SshapeX, typename SstrideX,  // x
          typename ScalarY, typename SshapeY, typename SstrideY,  // y
          typename = enable_if_same_scalar_t<ScalarX, ScalarY>>
MATHPRIM_NOINLINE Scalar dot(generic_type<ScalarX, SshapeX, SstrideX> x, generic_type<ScalarY, SshapeY, SstrideY> y) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS dot. (x)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                "Incompatible views for BLAS dot. (y)");

  MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::dot: x.shape() != y.shape()");

  return static_cast<Derived *>(this)->dot_impl(x.as_const().flatten(), y.as_const().flatten());
}

/**
  * @brief Compute the norm of x.
  *
  * @param x
  * @return Scalar
  */
template <typename ScalarX, typename Sshape, typename Sstride,  // x
          typename = enable_if_same_scalar_t<ScalarX>>
MATHPRIM_NOINLINE Scalar norm(generic_type<ScalarX, Sshape, Sstride> x) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS norm.");

  return static_cast<Derived *>(this)->norm_impl(x.as_const().flatten());
}

template <typename ScalarX, typename Sshape, typename Sstride,  // x
          typename = enable_if_same_scalar_t<ScalarX>>
MATHPRIM_NOINLINE Scalar asum(generic_type<ScalarX, Sshape, Sstride> x) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS asum.");

  return static_cast<Derived *>(this)->asum_impl(x.as_const().flatten());
}

/**
  * @brief Compute the index of the maximum element of x.
  *
  * @param x
  * @return index_t
  */
template <typename ScalarX, typename Sshape, typename Sstride,  // x
          typename = enable_if_same_scalar_t<ScalarX>>
MATHPRIM_NOINLINE index_t amax(generic_type<ScalarX, Sshape, Sstride> x) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS amax.");

  return static_cast<Derived *>(this)->amax_impl(x.as_const().flatten());
}

/**
  * @brief Computes Y <- X * Y, element-wise.
  * @param x
  * @return index_t
  */
template <typename ScalarX, typename SshapeX, typename SstrideX,  // x
          typename SshapeY, typename SstrideY,                    // y
          typename = enable_if_same_scalar_t<ScalarX>>
MATHPRIM_NOINLINE void emul(generic_type<ScalarX, SshapeX, SstrideX> x, view_type<SshapeY, SstrideY> y) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS emul. (x)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                "Incompatible views for BLAS emul. (y)");

  MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::emul: x.shape() != y.shape()");

  static_cast<Derived *>(this)->emul_impl(x.as_const().flatten(), y.flatten());
}

/**
  * @brief Computes Y <- alpha * A @ X + beta * Y
  *
  * @param alpha
  * @param mat_a if transpose, then A^T is used.
  * @param x
  * @param beta
  * @param y
  */
template <typename ScalarA, typename SshapeA, typename SstrideA,  // matrix A
          typename ScalarX, typename SshapeX, typename SstrideX,  // vector x
          typename SshapeY, typename SstrideY,                    // vector y
          typename = enable_if_same_scalar_t<ScalarA, ScalarX>>
MATHPRIM_NOINLINE void gemv(Scalar alpha, generic_type<ScalarA, SshapeA, SstrideA> mat_a,
                            generic_type<ScalarX, SshapeX, SstrideX> x, Scalar beta, view_type<SshapeY, SstrideY> y) {
  // check for shapes
  internal::check_mv_shapes(mat_a.shape(), x.shape(), y.shape());
  // check for capabilities
  auto mat_a_op = internal::get_matrix_op(mat_a);
  MATHPRIM_INTERNAL_CHECK_THROW(mat_a_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemv. (mat_a)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS gemv. (x)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                "Incompatible views for BLAS gemv. (y)");

  // Run the actual implementation
  static_cast<Derived *>(this)->gemv_impl(alpha, mat_a.as_const(), x.as_const().flatten(), beta, y.flatten());
}

/**
  * @brief Computes C <- alpha * A @ B + beta * C
  *
  * @param alpha
  * @param A if transpose, then A^T is used.
  * @param B if transpose, then B^T is used.
  * @param beta
  * @param C if transpose, then C^T is used.
  */
template <typename ScalarA, typename SshapeA, typename SstrideA,  // matrix A
          typename ScalarB, typename SshapeB, typename SstrideB,  // matrix B
          typename SshapeC, typename SstrideC,                    // matrix C
          typename = enable_if_same_scalar_t<ScalarA, ScalarB>>
MATHPRIM_NOINLINE void gemm(Scalar alpha, generic_type<ScalarA, SshapeA, SstrideA> A,
                            generic_type<ScalarB, SshapeB, SstrideB> B, Scalar beta, view_type<SshapeC, SstrideC> C) {
  // check for shapes
  internal::check_mm_shapes(A.shape(), B.shape(), C.shape());
  // check for capabilities
  auto a_op = internal::get_matrix_op(A);
  auto b_op = internal::get_matrix_op(B);
  auto c_op = internal::get_matrix_op(C);
  MATHPRIM_INTERNAL_CHECK_THROW(a_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemm. (A)");
  MATHPRIM_INTERNAL_CHECK_THROW(b_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemm. (B)");
  MATHPRIM_INTERNAL_CHECK_THROW(c_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemm. (C)");

  static_cast<Derived *>(this)->gemm_impl(alpha, A.as_const(), B.as_const(), beta, C);
}

template <typename ScalarA, typename SshapeA, typename SstrideA,  // batched matrix A
          typename ScalarB, typename SshapeB, typename SstrideB,  // batched matrix B
          typename SshapeC, typename SstrideC,                    // batched matrix C
          typename = enable_if_same_scalar_t<ScalarA, ScalarB>>
MATHPRIM_NOINLINE void gemm_batch_strided(Scalar alpha, generic_type<ScalarA, SshapeA, SstrideA> A,
                                          generic_type<ScalarB, SshapeB, SstrideB> B, Scalar beta,
                                          view_type<SshapeC, SstrideC> C) {
  // check for shapes
  internal::check_mm_shapes(A.slice(0).shape(), B.slice(0).shape(), C.slice(0).shape());
  // check for capabilities
  auto a_op = internal::get_matrix_op(A.slice(0));
  auto b_op = internal::get_matrix_op(B.slice(0));
  auto c_op = internal::get_matrix_op(C.slice(0));
  MATHPRIM_INTERNAL_CHECK_THROW(a_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemm. (A)");
  MATHPRIM_INTERNAL_CHECK_THROW(b_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemm. (B)");
  MATHPRIM_INTERNAL_CHECK_THROW(c_op != internal::matrix_op::invalid, std::runtime_error,
                                "Incompatible views for BLAS gemm. (C)");

  static_cast<Derived *>(this)->gemm_batch_strided_impl(alpha, A.as_const(), B.as_const(), beta, C);
}



///////////////////////////////////////////////////////////////////////////////
/// BLAS extensions.
///////////////////////////////////////////////////////////////////////////////
template <typename ScalarX, typename SshapeX, typename SstrideX,  // x
          typename SshapeY, typename SstrideY,  // y
          typename = enable_if_same_scalar_t<ScalarX, Scalar>>
MATHPRIM_NOINLINE void axpby(const Scalar &alpha, const generic_type<ScalarX, SshapeX, SstrideX> &x, const Scalar &beta, const view_type<SshapeY, SstrideY>& y) {
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(x), std::runtime_error,
                                "Incompatible views for BLAS axpby. (x)");
  MATHPRIM_INTERNAL_CHECK_THROW(internal::is_capable_vector(y), std::runtime_error,
                                "Incompatible views for BLAS axpby. (y)");
  MATHPRIM_INTERNAL_CHECK_THROW(x.shape() == y.shape(), shape_error, "blas::axpby: x.shape() != y.shape()");

  static_cast<Derived *>(this)->axpby_impl(alpha, x.as_const().flatten(), beta, y.flatten());
}
};

namespace internal {
template <typename Scalar, typename Device>
struct default_blas_selector;
}

template <typename Scalar, typename Device>
struct default_blas {
  using type = typename internal::default_blas_selector<Scalar, Device>::type;
};

template <typename Scalar, typename Device>
using default_blas_t = typename default_blas<Scalar, Device>::type;

}  // namespace blas
}  // namespace mathprim