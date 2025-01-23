/**
 * @brief Support for Eigen/Dense
 *   Greatly inspired by nanobind:
 *   https://github.com/wjakob/nanobind/blob/master/include/nanobind/eigen/dense.h
 */

#pragma once
#include "mathprim/core/defines.hpp"
#ifdef __CUDACC__
#  pragma nv_diagnostic push
#  pragma nv_diag_suppress 20012
#endif
#include <Eigen/Dense>
#ifdef __CUDACC__
#  pragma nv_diagnostic pop
#endif
#include <mathprim/core/common.hpp>        // IWYU pragma: export
#include <mathprim/core/utils/common.hpp>  // IWYU pragma: export

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#  error "EIGEN_DEFAULT_TO_ROW_MAJOR is not supported"
#endif

namespace mathprim::eigen_support {

namespace internal {

/********************************************************************
 * >>>> nanobind
 ********************************************************************/

/// Determine the number of dimensions of the given Eigen type
template <typename T> constexpr int ndim_v = bool(T::IsVectorAtCompileTime) ? 1 : 2;

/// Extract the compile-time strides of the given Eigen type
template <typename T> struct stride {
  using type = Eigen::Stride<0, 0>;
};

template <typename T, int Options, typename StrideType> struct stride<Eigen::Map<T, Options, StrideType>> {
  using type = StrideType;
};

template <typename T, int Options, typename StrideType> struct stride<Eigen::Ref<T, Options, StrideType>> {
  using type = StrideType;
};

template <typename T> using stride_t = typename stride<T>::type;

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
template <typename T> constexpr bool is_eigen_v = mathprim::internal::is_base_of_template_v<T, Eigen::EigenBase>;

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

template <typename T, device_t dev, int rows, int cols> constexpr Eigen::AlignmentType alignment_impl() {
  constexpr size_t device_align = buffer_backend_traits<dev>::alloc_alignment;
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
template <typename T, int rows, int cols> using matrix_t = internal::matrix_t<T, rows, cols>;
template <typename T, int rows> using vector_t = matrix_t<T, rows, 1>;

/// abbreviation for Eigen::Dynamic
constexpr int dynamic = Eigen::Dynamic;

/// determine the alignment of the Eigen matrix
template <typename T, device_t dev, int rows, int cols>
static constexpr Eigen::AlignmentType alignment_v = internal::alignment_impl<T, dev, rows, cols>();

/// Determine a proper type for mapped matrix
template <typename T, int rows, int cols, device_t dev>
using matrix_map_t = Eigen::Map<matrix_t<T, rows, cols>, Eigen::Unaligned, Eigen::Stride<dynamic, dynamic>>;

/// Determine a proper type for continuous mapped matrix
template <typename T, int rows, int cols, device_t dev>
using matrix_cmap_t = Eigen::Map<matrix_t<T, rows, cols>, alignment_v<T, dev, rows, cols>, Eigen::Stride<0, 0>>;

/// Determine a proper type for mapped vector
template <typename T, int rows, device_t dev>
using vector_map_t = Eigen::Map<vector_t<T, rows>, Eigen::Unaligned, Eigen::InnerStride<dynamic>>;

/// Determine a proper type for continuous mapped vector
template <typename T, int rows, device_t dev>
using vector_cmap_t = Eigen::Map<vector_t<T, rows>, alignment_v<T, dev, rows, 1>, Eigen::Stride<0, 0>>;

/**
 * @brief Create a continuous map to matrix from a buffer view.
 */
template <int rows = dynamic, int cols = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC matrix_cmap_t<T, rows, cols, dev> cmap(basic_view<T, 2, dev> view) {
  MATHPRIM_ASSERT(view.is_contiguous());
  const int dyn_rows = view.shape(0);
  if constexpr (rows != dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  const int dyn_cols = view.shape(1);
  if constexpr (cols != dynamic) {
    MATHPRIM_ASSERT(dyn_cols == cols);
  }
  return {view.data(), dyn_rows, dyn_cols};
}

/**
 * @brief Create a continuous map to vector from a buffer view.
 */
template <int rows = dynamic, typename T, device_t dev>
vector_cmap_t<T, rows, dev> MATHPRIM_PRIMFUNC cmap(basic_view<T, 1, dev> view) {
  MATHPRIM_ASSERT(view.is_contiguous());
  const int dyn_rows = view.shape(0);
  if constexpr (rows != dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  return {view.data(), dyn_rows, 1};
}

/**
 * @brief Create a map to matrix from a buffer view.
 */
template <int rows = dynamic, int cols = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC Eigen::Map<matrix_t<T, rows, cols>, Eigen::Unaligned, Eigen::Stride<dynamic, dynamic>> map(
    basic_view<T, 2, dev> view) {
  const int dyn_rows = view.shape(0);
  if constexpr (rows != dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  const int dyn_cols = view.shape(1);
  if constexpr (cols != dynamic) {
    MATHPRIM_ASSERT(dyn_cols == cols);
  }

  return {view.data(), dyn_rows, dyn_cols, Eigen::Stride<dynamic, dynamic>(view.stride(0), view.stride(1))};
}

/**
 * @brief Create a map to vector from a buffer view.
 */
template <int rows = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC Eigen::Map<vector_t<T, rows>, Eigen::Unaligned, Eigen::InnerStride<dynamic>> map(
    basic_view<T, 1, dev> view) {
  const int dyn_rows = view.shape(0);
  if constexpr (rows != dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  const int stride = view.stride(0);
  return {view.data(), dyn_rows, 1, Eigen::InnerStride<dynamic>(stride)};
}

/**
 * @brief Create a continuous Eigen::Ref from a buffer view. (matrix)
 */
template <int rows = dynamic, int cols = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC Eigen::Ref<matrix_t<T, rows, cols>, alignment_v<T, dev, rows, cols>> cref(
    basic_view<T, 2, dev> view) {
  MATHPRIM_ASSERT(view.is_contiguous());
  return {view.data(), view.shape(0), view.shape(1)};
}

/**
 * @brief Create a continuous Eigen::Ref from a buffer view. (vector)
 */
template <int rows = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC Eigen::Ref<vector_t<T, rows>, Eigen::Unaligned> cref(basic_view<T, 1, dev> view) {
  return {view.data(), view.shape(0), 1};
}

/**
 * @brief Create a Eigen::Ref from a buffer view. (matrix)
 */
template <int rows = dynamic, int cols = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC Eigen::Ref<matrix_t<T, rows, cols>, Eigen::Unaligned, Eigen::Stride<dynamic, dynamic>> ref(
    basic_view<T, 2, dev> view) {
  const int dyn_rows = view.shape(0);
  if constexpr (rows != dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  const int dyn_cols = view.shape(1);
  if constexpr (cols != dynamic) {
    MATHPRIM_ASSERT(dyn_cols == cols);
  }
  return {view.data(), dyn_rows, dyn_cols, Eigen::Stride<dynamic, dynamic>(view.stride(0), view.stride(1))};
}

/**
 * @brief Create a Eigen::Ref from a buffer view. (vector)
 */
template <int rows = dynamic, typename T, device_t dev>
MATHPRIM_PRIMFUNC Eigen::Ref<vector_t<T, rows>, Eigen::Unaligned, Eigen::InnerStride<dynamic>> ref(
    basic_view<T, 1, dev> view) {
  const int dyn_rows = view.shape(0);
  if constexpr (rows != dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  return {view.data(), dyn_rows, 1, Eigen::InnerStride<dynamic>(view.stride(0))};
}

}  // namespace mathprim::eigen_support
