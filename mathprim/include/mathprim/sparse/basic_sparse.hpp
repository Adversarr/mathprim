#pragma once
#include "mathprim/core/view.hpp"

namespace mathprim::sparse {

enum class sparse_format {
  csr,  // in Eigen, corresponding to compressed sparse row format
  csc,  // in Eigen, corresponding to compressed sparse column format
  coo,  // in coo format, we assume that the indices are sorted by row
  bsr,  // blocked compress row.
};

enum class sparse_property {
  general,
  symmetric,  // currently, we do not support symmetric uplo compression
  skew_symmetric,
  /* NOT SUPPORTED Part */
  // hermitian
};

template <typename Scalar, typename device, sparse_format sparse_compression>
class basic_sparse_view {
public:
  using values_view = continuous_view<Scalar, shape_t<keep_dim>, device>;
  static constexpr bool is_const = std::is_const_v<Scalar>;
  using index_type = std::conditional_t<is_const, const index_t, index_t>;
  using ptrs_view = continuous_view<std::conditional_t<is_const, const index_t, index_t>, shape_t<keep_dim>, device>;

  MATHPRIM_PRIMFUNC basic_sparse_view(Scalar* values, index_type* outer_ptrs, index_type* inner_indices, index_t rows,
                                      index_t cols, index_t nnz, sparse_property property, bool transpose) :
      basic_sparse_view(view<device>(values, make_shape(nnz)), view<device>(outer_ptrs, make_shape(rows + 1)),
                        view<device>(inner_indices, make_shape(nnz)), rows, cols, nnz, property, transpose) {}

  MATHPRIM_PRIMFUNC
  basic_sparse_view(values_view values, ptrs_view outer_ptrs, ptrs_view inner_indices, index_t rows, index_t cols,
                    index_t nnz, sparse_property property, bool transpose) :
      values_(values),
      outer_ptrs_(outer_ptrs),
      inner_indices_(inner_indices),
      rows_(rows),
      cols_(cols),
      nnz_(nnz),
      property_(property),
      is_transpose_(transpose) {
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

  MATHPRIM_PRIMFUNC bool is_transpose() const noexcept {
    return is_transpose_;
  }

  basic_sparse_view<std::add_const_t<Scalar>, device, sparse_compression> as_const() const noexcept {
    return basic_sparse_view<std::add_const_t<Scalar>, device, sparse_compression>(
        values_.as_const(), outer_ptrs_.as_const(), inner_indices_.as_const(), rows_, cols_, nnz_, property_,
        is_transpose_);
  }

private:
  values_view values_;
  ptrs_view outer_ptrs_;
  ptrs_view inner_indices_;

  index_t rows_;
  index_t cols_;
  index_t nnz_;
  sparse_property property_{sparse_property::general};
  bool is_transpose_{false};
};

// Sparse BLAS basic API.
template <typename Scalar, typename device, sparse_format sparse_compression>
class sparse_blas_base {
public:
  using vector_view = continuous_view<Scalar, shape_t<keep_dim>, device>;
  using const_vector_view = continuous_view<const Scalar, shape_t<keep_dim>, device>;
  using sparse_view = basic_sparse_view<Scalar, device, sparse_compression>;
  using const_sparse_view = basic_sparse_view<const Scalar, device, sparse_compression>;
  explicit sparse_blas_base(const_sparse_view matrix_view) : mat_(matrix_view) {}
  virtual ~sparse_blas_base() = default;

  // y = alpha * A * x + beta * y.
  virtual void gemv(Scalar alpha, const_vector_view x, Scalar beta, vector_view y) = 0;

  // // <x, y>_A = x^T * A * y.
  // virtual Scalar inner(const_vector_view x, const_vector_view y) = 0;

protected:
  void check_gemv_shape(const_vector_view x, vector_view y) const {
    auto [rows, cols] = mat_.shape();
    auto x_size = x.size();
    auto y_size = y.size();
    if (mat_.is_transpose()) {
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

}  // namespace mathprim::sparse
