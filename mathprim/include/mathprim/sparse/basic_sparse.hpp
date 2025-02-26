#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/view.hpp"
namespace mathprim::sparse {

enum class sparse_format {
  csr,  // in Eigen, corresponding to compressed sparse row format
  csc,  // in Eigen, corresponding to compressed sparse column format
  coo,  // in coo format, we assume that the indices are sorted by row
  /* Future works */
  // bsr,  // blocked compress row.
};

enum class sparse_property {
  general,
  symmetric,  // currently, we do not support symmetric uplo compression
  skew_symmetric,
  /* NOT SUPPORTED Part */
  // hermitian
};

template <typename Scalar>
struct sparse_entry {
  index_t row_;
  index_t col_;
  Scalar value_;

  sparse_entry(index_t row, index_t col, Scalar value) : row_(row), col_(col), value_(value) {}
};

template <typename Scalar, typename Device, sparse_format SparseCompression>
class basic_sparse_view {
public:
  using values_view = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  static constexpr bool is_const = std::is_const_v<Scalar>;
  using index_type = std::conditional_t<is_const, const index_t, index_t>;
  using ptrs_view = contiguous_view<std::conditional_t<is_const, const index_t, index_t>, shape_t<keep_dim>, Device>;

  static MATHPRIM_PRIMFUNC index_t outer_size(index_t rows, index_t cols, index_t nnz) {
    if constexpr (SparseCompression == sparse_format::csr) {
      return rows + 1;
    } else if constexpr (SparseCompression == sparse_format::csc) {
      return cols + 1;
    } else if constexpr (SparseCompression == sparse_format::coo) {
      return nnz;
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Not implemente");
      return 0;
    }
  }

  MATHPRIM_PRIMFUNC basic_sparse_view(Scalar* values, index_type* outer_ptrs, index_type* inner_indices, index_t rows,
                                      index_t cols, index_t nnz, sparse_property property) :
      basic_sparse_view(view<Device>(values, make_shape(nnz)),
                        view<Device>(outer_ptrs, make_shape(outer_size(rows, cols, nnz))),
                        view<Device>(inner_indices, make_shape(nnz)), rows, cols, nnz, property) {}

  basic_sparse_view() = default;
  basic_sparse_view(const basic_sparse_view&) = default;
  basic_sparse_view& operator=(const basic_sparse_view&) = default;

  MATHPRIM_PRIMFUNC
  basic_sparse_view(values_view values, ptrs_view outer_ptrs, ptrs_view inner_indices, index_t rows, index_t cols,
                    index_t nnz, sparse_property property) :
      values_(values),
      outer_ptrs_(outer_ptrs),
      inner_indices_(inner_indices),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      property_(property) {
    if (property_ == sparse_property::symmetric || property_ == sparse_property::skew_symmetric) {
      MATHPRIM_ASSERT(rows == cols && "Symmetric(or skew symmetric) matrix must be square.");
    }
  }

  MATHPRIM_PRIMFUNC values_view values() const noexcept {
    return values_;
  }

  MATHPRIM_PRIMFUNC ptrs_view outer_ptrs() const noexcept {
    return outer_ptrs_;
  }

  MATHPRIM_PRIMFUNC ptrs_view inner_indices() const noexcept {
    return inner_indices_;
  }

  MATHPRIM_PRIMFUNC index_t rows() const noexcept {
    return rows_;
  }

  MATHPRIM_PRIMFUNC index_t cols() const noexcept {
    return cols_;
  }

  MATHPRIM_PRIMFUNC dshape<2> shape() const noexcept {
    return dshape<2>(rows_, cols_);
  }

  MATHPRIM_PRIMFUNC index_t nnz() const noexcept {
    return nnz_;
  }

  MATHPRIM_PRIMFUNC sparse_property property() const noexcept {
    return property_;
  }

  basic_sparse_view<std::add_const_t<Scalar>, Device, SparseCompression> as_const() const noexcept {
    return basic_sparse_view<std::add_const_t<Scalar>, Device, SparseCompression>(
        values_.as_const(), outer_ptrs_.as_const(), inner_indices_.as_const(), rows_, cols_, nnz_, property_);
  }

private:
  values_view values_;
  ptrs_view outer_ptrs_;
  ptrs_view inner_indices_;

  index_t rows_{0};
  index_t cols_{0};
  index_t nnz_{0};
  sparse_property property_{sparse_property::general};
};

template <typename Scalar, typename Device, sparse_format SparseCompression>
class basic_sparse_matrix {
  static inline index_t outer_size(index_t rows, index_t cols, index_t nnz) {
    if constexpr (SparseCompression == sparse_format::csr) {
      return rows + 1;
    } else if constexpr (SparseCompression == sparse_format::csc) {
      return cols + 1;
    } else if constexpr (SparseCompression == sparse_format::coo) {
      return nnz;
    } else {
      static_assert(::mathprim::internal::always_false_v<Scalar>, "Not implemente");
      return 0;
    }
  }

public:
  using values_buffer = contiguous_buffer<Scalar, dshape<1>, Device>;
  using index_buffer = contiguous_buffer<index_t, dshape<1>, Device>;
  using view_type = basic_sparse_view<Scalar, Device, SparseCompression>;
  using const_view_type = basic_sparse_view<const Scalar, Device, SparseCompression>;
  basic_sparse_matrix() = default;

  basic_sparse_matrix(index_t rows, index_t cols, index_t nnz, sparse_property property = sparse_property::general) :
      values_(make_buffer<Scalar, Device>(nnz)),
      outer_ptrs_(make_buffer<index_t, Device>(outer_size(rows, cols, nnz))),
      inner_indices_(make_buffer<index_t, Device>(nnz)),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      property_(property) {}

  basic_sparse_matrix(values_buffer values, index_buffer outer_ptrs, index_buffer inner_indices, index_t rows,
                      index_t cols, index_t nnz, sparse_property property = sparse_property::general) :
      values_(std::move(values)),
      outer_ptrs_(std::move(outer_ptrs)),
      inner_indices_(std::move(inner_indices)),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      property_(property) {
    if constexpr (SparseCompression == sparse_format::csr) {
      MATHPRIM_INTERNAL_CHECK_THROW(outer_ptrs_.size() == rows + 1, shape_error,
                                    "The size of outer_ptrs is not equal to rows + 1.");
    } else if constexpr (SparseCompression == sparse_format::csc) {
      MATHPRIM_INTERNAL_CHECK_THROW(outer_ptrs_.size() == cols + 1, shape_error,
                                    "The size of outer_ptrs is not equal to cols + 1.");
    } else if constexpr (SparseCompression == sparse_format::coo) {
      MATHPRIM_INTERNAL_CHECK_THROW(outer_ptrs_.size() == nnz, shape_error,
                                    "The size of outer_ptrs is not equal to nnz.");
    }

    MATHPRIM_INTERNAL_CHECK_THROW(inner_indices_.size() == nnz, shape_error,
                                  "The size of inner_indices is not equal to nnz.");
    MATHPRIM_INTERNAL_CHECK_THROW(values_.size() == nnz, shape_error, "The size of values is not equal to nnz.");
  }

  basic_sparse_matrix(const basic_sparse_matrix& other) = delete;
  basic_sparse_matrix& operator=(const basic_sparse_matrix& other) = delete;
  basic_sparse_matrix(basic_sparse_matrix&& other) = default;
  basic_sparse_matrix& operator=(basic_sparse_matrix&& other) = default;

  MATHPRIM_FORCE_INLINE view_type view() noexcept {
    return view_type(values_.view(), outer_ptrs_.view(), inner_indices_.view(), rows_, cols_, nnz_, property_);
  }

  MATHPRIM_FORCE_INLINE const_view_type view() const noexcept {
    return const_view();
  }

  MATHPRIM_FORCE_INLINE const_view_type const_view() const noexcept {
    return const_view_type(values_.view(), outer_ptrs_.view(), inner_indices_.view(), rows_, cols_, nnz_, property_);
  }

  MATHPRIM_FORCE_INLINE values_buffer& values() noexcept {
    return values_;
  }

  MATHPRIM_FORCE_INLINE index_buffer& outer_ptrs() noexcept {
    return outer_ptrs_;
  }

  MATHPRIM_FORCE_INLINE index_buffer& inner_indices() noexcept {
    return inner_indices_;
  }

  MATHPRIM_FORCE_INLINE index_t rows() const noexcept {
    return rows_;
  }

  MATHPRIM_FORCE_INLINE index_t cols() const noexcept {
    return cols_;
  }

  MATHPRIM_FORCE_INLINE index_t nnz() const noexcept {
    return nnz_;
  }

  MATHPRIM_FORCE_INLINE sparse_property property() const noexcept {
    return property_;
  }

  template <typename Device2>
  basic_sparse_matrix<Scalar, Device2, SparseCompression> to(Device2 = {}) {
    auto value_buf = values_.template to<Device2>();
    auto outer_buf = outer_ptrs_.template to<Device2>();
    auto inner_buf = inner_indices_.template to<Device2>();
    return {std::move(value_buf), std::move(outer_buf), std::move(inner_buf), rows_, cols_, nnz_, property_};
  }

protected:
  values_buffer values_;
  index_buffer outer_ptrs_;
  index_buffer inner_indices_;
  index_t rows_{0};
  index_t cols_{0};
  index_t nnz_{0};
  sparse_property property_{sparse_property::general};
};

// Sparse BLAS basic API.
template <typename Derived, typename Scalar, typename Device, sparse_format SparseCompression>
class sparse_blas_base {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  static constexpr sparse_format compression = SparseCompression;
  using vector_view = contiguous_view<Scalar, shape_t<keep_dim>, Device>;
  using const_vector_view = contiguous_view<const Scalar, shape_t<keep_dim>, Device>;
  using sparse_view = basic_sparse_view<Scalar, Device, SparseCompression>;
  using const_sparse_view = basic_sparse_view<const Scalar, Device, SparseCompression>;
  explicit sparse_blas_base(const_sparse_view matrix_view) : mat_(matrix_view) {}
  virtual ~sparse_blas_base() = default;

  const_sparse_view matrix() const noexcept {
    return mat_;
  }

  // y = alpha * A * x + beta * y.
  void gemv(Scalar alpha, const_vector_view x, Scalar beta, vector_view y, bool transpose = false) {
    check_gemv_shape(x, y, transpose);
    static_cast<Derived*>(this)->gemv_impl(alpha, x, beta, y, transpose);
  }

  // // <x, y>_A = x^T * A * y.
  // virtual Scalar inner(const_vector_view x, const_vector_view y) = 0;

protected:
  void check_gemv_shape(const_vector_view x, vector_view y, bool transpose) const {
    auto [rows, cols] = mat_.shape();
    auto x_size = x.size();
    auto y_size = y.size();
    if (transpose) {
      // [cols, rows] * [rows] = [cols].
      if (rows != x_size) {
        throw std::runtime_error("The size of x is not equal to the number of rows of the matrix.");
      }
      if (cols != y_size) {
        throw std::runtime_error("The size of y is not equal to the number of cols of the matrix.");
      }
    } else {
      // [rows, cols] * [cols] = [rows].
      if (cols != x_size) {
        throw std::runtime_error("The size of x is not equal to the number of cols of the matrix.");
      }
      if (rows != y_size) {
        throw std::runtime_error("The size of y is not equal to the number of rows of the matrix.");
      }
    }
  }

  // NOTE: Store the matrix view is necessary for descriptors.
  const_sparse_view mat_;  ///< The sparse matrix view.
};

/**
 * @brief Supports for sparse entry visitor
 * 
 */
template <typename F, typename Parallel, typename Scalar, sparse_format SparseCompression>
void visit(const basic_sparse_view<Scalar, device::cpu, SparseCompression>& view, Parallel&& par, F&& fn) {
  auto outer = view.outer_ptrs();
  auto inner = view.inner_indices();
  auto values = view.values();

  if constexpr (SparseCompression == sparse_format::csr) {
    par.run(make_shape(view.rows()), [&] (index_t i) {
      auto start = outer[i];
      auto end = outer[i + 1];
      for (index_t j = start; j < end; ++j) {
        fn(i, inner[j], values[j]);
      }
    });
  } else if constexpr (SparseCompression == sparse_format::csc) {
    par.run(make_shape(view.cols()), [&] (index_t j) {
      auto start = outer[j];
      auto end = outer[j + 1];
      for (index_t i = start; i < end; ++i) {
        fn(inner[i], j, values[i]);
      }
    });
  } else if constexpr (SparseCompression == sparse_format::coo) {
    par.run(make_shape(view.nnz()), [&] (index_t i) {
      fn(inner[i], outer[i], values[i]);
    });
  } else {
    static_assert(::mathprim::internal::always_false_v<Scalar>, "Not implemented");
  }
}

#ifdef __CUDACC__
/**
 * @brief Supports for sparse entry visitor
 * 
 */
template <typename F, typename Parallel, typename Scalar, sparse_format SparseCompression>
void visit(const basic_sparse_view<Scalar, device::cuda, SparseCompression>& view, Parallel&& cuda, F&& fn) {
  auto outer = view.outer_ptrs();
  auto inner = view.inner_indices();
  auto values = view.values();

  if constexpr (SparseCompression == sparse_format::csr) {
    auto rows = view.rows();
    cuda.run(make_shape(rows), [outer = outer.data(), inner = inner.data(), values = values.data(), fn] __device__ (index_t i) {
      auto start = outer[i];
      auto end = outer[i + 1];
      for (index_t j = start; j < end; ++j) {
        fn(i, inner[j], values[j]);
      }
    });
  } else if constexpr (SparseCompression == sparse_format::csc) {
    auto cols = view.cols();
    cuda.run(make_shape(cols), [outer = outer.data(), inner = inner.data(), values = values.data(), fn] __device__ (index_t j) {
      auto start = outer[j];
      auto end = outer[j + 1];
      for (index_t i = start; i < end; ++i) {
        fn(inner[i], j, values[i]);
      }
    });
  } else if constexpr (SparseCompression == sparse_format::coo) {
    auto nnz = view.nnz();
    cuda.run(make_shape(nnz), [outer = outer.data(), inner = inner.data(), values = values.data(), fn] __device__ (index_t i) {
      fn(inner[i], outer[i], values[i]);
    });
  } else {
    static_assert(::mathprim::internal::always_false_v<Scalar>, "Not implemented");
  }
}
#endif

}  // namespace mathprim::sparse
