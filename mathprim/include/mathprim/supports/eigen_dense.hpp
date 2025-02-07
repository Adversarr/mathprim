/**
 * @brief Support for Eigen/Dense
 *   Greatly inspired by nanobind:
 *   https://github.com/wjakob/nanobind/blob/master/include/nanobind/eigen/dense.h
 */

#pragma once

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
#include "mathprim/core/utils/common.hpp"
#ifdef __CUDACC__
#  pragma nv_diagnostic push
#  pragma nv_diag_suppress 20012
#endif
#include <Eigen/Dense>
#ifdef __CUDACC__
#  pragma nv_diagnostic pop
#endif
#include "mathprim/blas/blas.hpp"

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#  error "EIGEN_DEFAULT_TO_ROW_MAJOR is not supported"
#endif

namespace mathprim::eigen_support {

namespace internal {

/********************************************************************
 * >>>> nanobind
 ********************************************************************/

/// Determine the number of dimensions of the given Eigen type
template <typename T>
constexpr int ndim_v = bool(T::IsVectorAtCompileTime) ? 1 : 2;

/// Extract the compile-time strides of the given Eigen type
template <typename T>
struct stride {
  using type = Eigen::Stride<0, 0>;
};

template <typename T, int Options, typename StrideType>
struct stride<Eigen::Map<T, Options, StrideType>> {
  using type = StrideType;
};

template <typename T, int Options, typename StrideType>
struct stride<Eigen::Ref<T, Options, StrideType>> {
  using type = StrideType;
};

template <typename T>
using stride_t = typename stride<T>::type;

/** \brief Identify types with a contiguous memory representation.
 *
 * This includes all specializations of ``Eigen::Matrix``/``Eigen::Array`` and
 * certain specializations of ``Eigen::Map`` and ``Eigen::Ref``. Note: Eigen
 * interprets a compile-time stride of 0 as contiguous.
 */
template <typename T>
constexpr bool is_contiguous_v
    = (stride_t<T>::InnerStrideAtCompileTime == 0 || stride_t<T>::InnerStrideAtCompileTime == 1)
      && (ndim_v<T> == 1 || stride_t<T>::OuterStrideAtCompileTime == 0
          || (stride_t<T>::OuterStrideAtCompileTime != Eigen::Dynamic
              && int(stride_t<T>::OuterStrideAtCompileTime) == int(T::InnerSizeAtCompileTime)));

/// Identify types with a static or dynamic layout that support contiguous
/// storage
template <typename T>
constexpr bool can_map_contiguous_memory_v
    = (stride_t<T>::InnerStrideAtCompileTime == 0 || stride_t<T>::InnerStrideAtCompileTime == 1
       || stride_t<T>::InnerStrideAtCompileTime == Eigen::Dynamic)
      && (ndim_v<T> == 1 || stride_t<T>::OuterStrideAtCompileTime == 0
          || stride_t<T>::OuterStrideAtCompileTime == Eigen::Dynamic
          || int(stride_t<T>::OuterStrideAtCompileTime) == int(T::InnerSizeAtCompileTime));

/// Any kind of Eigen class
template <typename T>
constexpr bool is_eigen_v = mathprim::internal::is_base_of_template_v<T, Eigen::EigenBase>;

/// Detects Eigen::Array, Eigen::Matrix, etc.
template <typename T>
constexpr bool is_eigen_plain_v = mathprim::internal::is_base_of_template_v<T, Eigen::PlainObjectBase>;

/// Detect Eigen::SparseMatrix
template <typename T>
constexpr bool is_eigen_sparse_v = mathprim::internal::is_base_of_template_v<T, Eigen::SparseMatrixBase>;

/// Detects expression templates
template <typename T>
constexpr bool is_eigen_xpr_v = is_eigen_v<T> && !is_eigen_plain_v<T> && !is_eigen_sparse_v<T>
                                && !std::is_base_of_v<Eigen::MapBase<T, Eigen::ReadOnlyAccessors>, T>;

/********************************************************************
 * <<<< nanobind
 ********************************************************************/

constexpr Eigen::AlignmentType get_eigen_alignment(size_t alignment) {
  if (alignment == 0) {
    return Eigen::Unaligned;
  } else if (alignment % 128 == 0) {
    return Eigen::Aligned128;
  } else if (alignment % 64 == 0) {
    return Eigen::Aligned64;
  } else if (alignment % 32 == 0) {
    return Eigen::Aligned32;
  } else if (alignment % 16 == 0) {
    return Eigen::Aligned16;
  } else if (alignment % 8 == 0) {
    return Eigen::Aligned8;
  }
  return Eigen::Unaligned;
}

template <typename T, class dev, int rows, int cols>
constexpr Eigen::AlignmentType alignment_impl() {
  constexpr size_t device_align = device::device_traits<dev>::alloc_alignment;
  constexpr size_t bytes = sizeof(T) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  constexpr bool can_align = get_eigen_alignment(bytes) != Eigen::Unaligned;

  return can_align ? get_eigen_alignment(device_align) : Eigen::Unaligned;
}

template <int prefer_major, int rows, int cols>
constexpr int determine_major_v = ((rows == 1 && cols != 1)   ? Eigen::RowMajor
                                   : (cols == 1 && rows != 1) ? Eigen::ColMajor
                                                              : prefer_major);

template <typename T, int rows, int cols, int prefer_major = Eigen::ColMajor, int align_en = Eigen::AutoAlign>
using matrix_t = std::conditional_t<
    !std::is_const_v<T>, Eigen::Matrix<T, rows, cols, (align_en | determine_major_v<prefer_major, rows, cols>)>,
    const Eigen::Matrix<std::remove_const_t<T>, rows, cols, (align_en | determine_major_v<prefer_major, rows, cols>)>>;

}  // namespace internal

/// Eigen matrix type
template <typename T, int rows, int cols>
using matrix_t = internal::matrix_t<T, rows, cols>;
template <typename T, int rows>
using vector_t = matrix_t<T, rows, 1>;

/// abbreviation for Eigen::Dynamic
constexpr int dynamic = Eigen::Dynamic;
template <index_t val>
constexpr int to_eigen_v = val == keep_dim ? Eigen::Dynamic : static_cast<int>(val);
template <index_t val>
constexpr int from_eigen_v = val == Eigen::Dynamic ? keep_dim : static_cast<index_t>(val);

using Index = Eigen::Index;

MATHPRIM_CONSTEXPR MATHPRIM_PRIMFUNC Index to_eigen_index(index_t idx) noexcept {
  return static_cast<Index>(idx);
}

/// determine the alignment of the Eigen matrix
template <typename T, typename dev, int rows, int cols>
static constexpr Eigen::AlignmentType alignment_v = internal::alignment_impl<T, dev, rows, cols>();

/// Determine a proper type for mapped matrix
template <typename T, int rows, int cols, typename dev>
using matrix_map_t = Eigen::Map<matrix_t<T, cols, rows>, Eigen::Unaligned, Eigen::Stride<dynamic, dynamic>>;

/// Determine a proper type for continuous mapped matrix
template <typename T, int rows, int cols, typename dev>
using matrix_cmap_t = Eigen::Map<matrix_t<T, cols, rows>, alignment_v<T, dev, rows, cols>, Eigen::Stride<0, 0>>;

template <typename EigenMatrix>
using from_eigen_shape_t = std::conditional_t<
    EigenMatrix::ColsAtCompileTime == 1, shape_t<from_eigen_v<EigenMatrix::RowsAtCompileTime>>,
    shape_t<from_eigen_v<EigenMatrix::ColsAtCompileTime>, from_eigen_v<EigenMatrix::RowsAtCompileTime>>>;

template <typename EigenMatrix, bool is_const, typename device>
using matrix_view_t
    = basic_view<std::conditional_t<is_const, const typename EigenMatrix::Scalar, typename EigenMatrix::Scalar>,
                 from_eigen_shape_t<EigenMatrix>,
                 ::mathprim::default_stride_t<typename EigenMatrix::Scalar, from_eigen_shape_t<EigenMatrix>>, device>;

/// Determine a proper type for mapped vector
template <typename T, int rows, typename dev>
using vector_map_t = Eigen::Map<vector_t<T, rows>, Eigen::Unaligned, Eigen::InnerStride<dynamic>>;

/// Determine a proper type for continuous mapped vector
template <typename T, int rows, typename dev>
using vector_cmap_t = Eigen::Map<vector_t<T, rows>, alignment_v<T, dev, rows, 1>, Eigen::Stride<0, 0>>;

/**
 * @brief Create a continuous map to matrix from a buffer view.
 */
template <typename Scalar, index_t s_rows, index_t s_cols, index_t outer_stride, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC matrix_cmap_t<Scalar, to_eigen_v<s_rows>, to_eigen_v<s_cols>, dev> cmap(
    basic_view<Scalar, shape_t<s_rows, s_cols>, stride_t<outer_stride, inner_stride>, dev> view) noexcept {
  MATHPRIM_ASSERT(view.is_contiguous());
  auto [cols, rows] = view.shape();
  return {view.data(), rows, cols};
}

/**
 * @brief Create a continuous map to vector from a buffer view.
 */
template <typename Scalar, index_t s_rows, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC vector_cmap_t<Scalar, to_eigen_v<s_rows>, dev> cmap(
    basic_view<Scalar, shape_t<s_rows>, stride_t<inner_stride>, dev> view) noexcept {
  MATHPRIM_ASSERT(view.is_contiguous());
  return {view.data(), view.shape(0)};
}

template <typename dev = device::cpu, typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
MATHPRIM_PRIMFUNC matrix_view_t<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, true, dev> view(
    const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &mat) noexcept {
  using ret = matrix_view_t<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, true, dev>;
  if constexpr (Cols == 1) {
    return ret{mat.data(), typename ret::sshape{mat.size()}};
  } else {
    return ret{mat.data(), typename ret::sshape{mat.cols(), mat.rows()}};
  }
}

/**
 * @brief Create a map to matrix from a buffer view.
 */
template <typename Scalar, index_t s_rows, index_t s_cols, index_t outer_stride, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC matrix_map_t<Scalar, to_eigen_v<s_rows>, to_eigen_v<s_cols>, dev> map(
    basic_view<Scalar, shape_t<s_rows, s_cols>, stride_t<outer_stride, inner_stride>, dev> view) noexcept {
  auto [cols, rows] = view.shape();
  auto [outer, inner] = view.stride();
  MATHPRIM_ASSERT(outer % sizeof(Scalar) == 0);
  MATHPRIM_ASSERT(inner % sizeof(Scalar) == 0);
  return {view.data(), rows, cols,
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(outer / sizeof(Scalar), inner / sizeof(Scalar))};
}

template <typename Scalar, index_t s_rows, index_t s_cols, index_t outer_stride, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC std::conditional_t<::mathprim::internal::is_continuous_compile_time_v<
                                         Scalar, shape_t<s_rows, s_cols>, stride_t<outer_stride, inner_stride>>,
                                     matrix_cmap_t<Scalar, to_eigen_v<s_rows>, to_eigen_v<s_cols>, dev>,
                                     matrix_map_t<Scalar, to_eigen_v<s_rows>, to_eigen_v<s_cols>, dev>>
amap(basic_view<Scalar, shape_t<s_rows, s_cols>, stride_t<outer_stride, inner_stride>, dev> view) noexcept {
  if constexpr (::mathprim::internal::is_continuous_compile_time_v<Scalar, shape_t<s_rows, s_cols>,
                                                                   stride_t<outer_stride, inner_stride>>) {
    return cmap<Scalar, s_rows, s_cols, outer_stride, inner_stride, dev>(view);
  } else {
    return map<Scalar, s_rows, s_cols, outer_stride, inner_stride, dev>(view);
  }
}

template <typename Scalar, index_t s_rows, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC std::conditional_t<
    ::mathprim::internal::is_continuous_compile_time_v<Scalar, shape_t<s_rows>, stride_t<inner_stride>>,
    vector_cmap_t<Scalar, to_eigen_v<s_rows>, dev>, vector_map_t<Scalar, to_eigen_v<s_rows>, dev>>
amap(basic_view<Scalar, shape_t<s_rows>, stride_t<inner_stride>, dev> view) noexcept {
  if constexpr (::mathprim::internal::is_continuous_compile_time_v<Scalar, shape_t<s_rows>, stride_t<inner_stride>>) {
    return cmap<Scalar, s_rows, inner_stride, dev>(view);
  } else {
    return map<Scalar, s_rows, inner_stride, dev>(view);
  }
}

/**
 * @brief Create a map to vector from a buffer view.
 */
template <typename Scalar, index_t s_rows, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC vector_map_t<Scalar, to_eigen_v<s_rows>, dev> map(
    basic_view<Scalar, shape_t<s_rows>, stride_t<inner_stride>, dev> view) noexcept {
  auto [rows] = view.shape();
  auto [inner] = view.stride();
  MATHPRIM_ASSERT(inner % sizeof(Scalar) == 0);
  return {view.data(), rows, Eigen::InnerStride<Eigen::Dynamic>(inner / sizeof(Scalar))};
}

/**
 * @brief Create a continuous Eigen::Ref from a buffer view. (matrix)
 */
template <typename Scalar, index_t s_rows, index_t s_cols, index_t outer_stride, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC Eigen::Ref<matrix_t<Scalar, to_eigen_v<s_rows>, to_eigen_v<s_cols>>,
                             alignment_v<Scalar, dev, to_eigen_v<s_rows>, to_eigen_v<s_cols>>>
cref(basic_view<Scalar, shape_t<s_rows, s_cols>, stride_t<outer_stride, inner_stride>, dev> view) noexcept {
  MATHPRIM_ASSERT(view.is_contiguous());
  auto [rows, cols] = view.shape();
  return {view.data(), rows, cols};
}

/**
 * @brief Create a continuous Eigen::Ref from a buffer view. (vector)
 */
template <typename Scalar, index_t s_rows, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC Eigen::Ref<vector_t<Scalar, to_eigen_v<s_rows>>, alignment_v<Scalar, dev, to_eigen_v<s_rows>, 1>>
cref(basic_view<Scalar, shape_t<s_rows>, stride_t<inner_stride>, dev> view) noexcept {
  MATHPRIM_ASSERT(view.is_contiguous());
  return {view.data(), view.shape(0)};
}

/**
 * @brief Create a Eigen::Ref from a buffer view. (matrix)
 */
template <typename Scalar, index_t s_rows, index_t s_cols, index_t outer_stride, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC Eigen::Ref<matrix_t<Scalar, to_eigen_v<s_rows>, to_eigen_v<s_cols>>,
                             alignment_v<Scalar, dev, to_eigen_v<s_rows>, to_eigen_v<s_cols>>>
ref(basic_view<Scalar, shape_t<s_rows, s_cols>, stride_t<outer_stride, inner_stride>, dev> view) noexcept {
  auto [rows, cols] = view.shape();
  auto [outer, inner] = view.stride();
  MATHPRIM_ASSERT(outer % sizeof(Scalar) == 0);
  MATHPRIM_ASSERT(inner % sizeof(Scalar) == 0);
  return {view.data(), rows, cols,
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(outer / sizeof(Scalar), inner / sizeof(Scalar))};
}

/**
 * @brief Create a Eigen::Ref from a buffer view. (vector)
 */
template <typename Scalar, index_t s_rows, index_t inner_stride, typename dev>
MATHPRIM_PRIMFUNC Eigen::Ref<vector_t<Scalar, to_eigen_v<s_rows>>, alignment_v<Scalar, dev, to_eigen_v<s_rows>, 1>> ref(
    basic_view<Scalar, shape_t<s_rows>, stride_t<inner_stride>, dev> view) noexcept {
  auto [rows] = view.shape();
  auto [inner] = view.stride();
  MATHPRIM_ASSERT(inner % sizeof(Scalar) == 0);
  return {view.data(), rows, Eigen::InnerStride<Eigen::Dynamic>(inner / sizeof(Scalar))};
}

}  // namespace mathprim::eigen_support
