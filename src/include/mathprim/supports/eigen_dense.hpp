#pragma once
// if you want to force column major, define
//      MATHPRIM_SUPPORT_EIGEN_FORCE_COLUMN_MAJOR,
// otherwise, it is your duty to guarantee your data accessing is correct.
#ifdef MATHPRIM_SUPPORT_EIGEN_FORCE_ROW_MAJOR
#  define EIGEN_DEFAULT_TO_ROW_MAJOR
#endif

#ifdef __CUDACC__
#  pragma nv_diagnostic push
#  pragma nv_diag_suppress 20012
#endif
#include <eigen3/Eigen/Dense>
#ifdef __CUDACC__
#  pragma nv_diagnostic pop
#endif
#include <mathprim/core/common.hpp>  // IWYU pragma: export

namespace mathprim::eigen_support {

namespace internal {

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

template <typename T, device_t dev, int rows, int cols>
struct continuous_buffer_alignment {
  // Default is not aligned.
  static constexpr size_t alloc_alignment
      = buffer_backend_traits<dev>::alloc_alignment;

  static constexpr Eigen::AlignmentType value
      = get_eigen_alignment(sizeof(T) * static_cast<size_t>(rows)
                            * static_cast<size_t>(cols))
                == Eigen::Unaligned
            ? Eigen::Unaligned
            : get_eigen_alignment(alloc_alignment);
};

template <typename T, int rows, int cols>
using eigen_type = std::conditional_t<
    std::is_const_v<T>, const Eigen::Matrix<std::remove_const_t<T>, rows, cols>,
    Eigen::Matrix<T, rows, cols>>;

}  // namespace internal

// Import Eigen classes here.
using Eigen::Matrix, Eigen::Map, Eigen::Vector;

/**
 * @brief Create a continuous map from a buffer view.
 *
 * @tparam T
 * @tparam dev
 * @tparam rows
 * @tparam cols
 */
template <typename T, device_t dev, int rows, int cols>
static constexpr Eigen::AlignmentType continuous_buffer_alignment_v
    = internal::continuous_buffer_alignment<T, dev, rows, cols>::value;

template <int rows = Eigen::Dynamic, int cols = Eigen::Dynamic, typename T,
          device_t dev>
Eigen::Map<internal::eigen_type<T, rows, cols>,
           continuous_buffer_alignment_v<T, dev, rows, cols>>
cmap(basic_buffer_view<T, 2, dev> view) {
  MATHPRIM_ASSERT(view.is_contiguous());
  const int dyn_rows = view.shape(0);
  if constexpr (rows != Eigen::Dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  const int dyn_cols = view.shape(1);
  if constexpr (cols != Eigen::Dynamic) {
    MATHPRIM_ASSERT(dyn_cols == cols);
  }
  return {view.data(), dyn_rows, dyn_cols};
}

template <int rows = Eigen::Dynamic, int cols = Eigen::Dynamic, typename T,
          device_t dev>
Eigen::Map<internal::eigen_type<T, rows, cols>,
           continuous_buffer_alignment_v<T, dev, rows, cols>,
           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
map(basic_buffer_view<T, 2, dev> view) {
  const int dyn_rows = view.shape(0);
  if constexpr (rows != Eigen::Dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  const int dyn_cols = view.shape(1);
  if constexpr (cols != Eigen::Dynamic) {
    MATHPRIM_ASSERT(dyn_cols == cols);
  }

  return {view.data(), dyn_rows, dyn_cols,
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(view.stride(0),
                                                        view.stride(1))};
}

template <int rows = Eigen::Dynamic, typename T, device_t dev>
Eigen::Map<internal::eigen_type<T, rows, 1>,
           continuous_buffer_alignment_v<T, dev, rows, 1>>
cmap(basic_buffer_view<T, 1, dev> view) {
  MATHPRIM_ASSERT(view.is_contiguous());
  const int dyn_rows = view.shape(0);
  if constexpr (rows != Eigen::Dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  return {view.data(), dyn_rows};
}

template <int rows = Eigen::Dynamic, typename T, device_t dev>
Eigen::Map<internal::eigen_type<T, rows, 1>,
           continuous_buffer_alignment_v<T, dev, rows, 1>,
           Eigen::InnerStride<Eigen::Dynamic>>
map(basic_buffer_view<T, 1, dev> view) {
  const int dyn_rows = view.shape(0);
  if constexpr (rows != Eigen::Dynamic) {
    MATHPRIM_ASSERT(dyn_rows == rows);
  }
  return {view.data(), dyn_rows,
          Eigen::InnerStride<Eigen::Dynamic>(view.stride(0))};
}

}  // namespace mathprim::eigen_support
