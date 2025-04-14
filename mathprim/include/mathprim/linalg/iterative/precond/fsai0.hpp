#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "mathprim/core/buffer.hpp"
#include "mathprim/linalg/iterative/iterative.hpp"

namespace mathprim::sparse::iterative {

namespace internal {
template <typename Scalar>
void fsai_compute(sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr> A,
                  sparse::basic_sparse_view<Scalar, device::cpu, sparse::sparse_format::csr> lo) {
  // A and lo have same structure(for lower triangular).
  // we fill in the approx inverse of A in lo, s.t. lo lo.T = A^-1.
  auto row_ptr_in = A.outer_ptrs();
  auto col_ind_in = A.inner_indices();
  auto row_ptr_out = lo.outer_ptrs();
  auto col_ind_out = lo.inner_indices();
  auto a_values = A.values();
  auto lo_values = lo.values();

  auto find_original_value = [&](index_t row, index_t col) -> Scalar {
    auto row_start = row_ptr_in[row];
    auto row_end = row_ptr_in[row + 1];
    for (auto i = row_start; i < row_end; ++i) {
      if (col_ind_in[i] == col) {
        return a_values[i];
      }
    }
    return 0;  // not found indicates zero.
  };

  auto job_of_row = [&](index_t row) mutable {
    const auto row_start = static_cast<size_t>(row_ptr_out(row));
    const index_t row_size = row_ptr_out(row + 1) - row_ptr_out(row);
    using DoubleMatrix = Eigen::MatrixX<double>;
    using DoubleVector = Eigen::Vector<double, Eigen::Dynamic>;

    DoubleMatrix mat(row_size, row_size);
    mat.setIdentity();
    mat *= static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    // DoubleVector b = DoubleVector::Unit(row_size, row_size - 1);
    DoubleVector b = DoubleVector::Zero(row_size);
    b(row_size - 1) = 1;

    for (int j = 0; j < row_size; ++j) {
      auto g_j_coresp_col = col_ind_out[row_start + j];
      for (int i = 0; i < row_size; ++i) {
        // for (i, j) find the corresponding coefficient in original matrix.
        auto g_i_coresp_col = col_ind_out[row_start + i];
        auto a_ij = find_original_value(g_j_coresp_col, g_i_coresp_col);
        mat(i, j) += static_cast<double>(a_ij);
      }
    }

    // solve the linear system.
    Scalar& row_start_value = lo_values[row_start];
    Eigen::Map<Eigen::Vector<Scalar, Eigen::Dynamic>> x(&row_start_value, row_size);
    auto llt = mat.ldlt();
    if (llt.info() != Eigen::Success) {
      throw std::runtime_error("Submatrix cholesky decomposition failed.");
    }
    Eigen::VectorXd x_double = llt.solve(b);
    double x_last = x_double(row_size - 1);
    x_double /= (::std::sqrt(x_last) + std::numeric_limits<double>::epsilon());
    x = x_double.cast<Scalar>();
  };

#if MATHPRIM_ENABLE_OPENMP
#  pragma omp parallel for schedule(dynamic) if (lo.nnz() > 100000)
#endif
  for (index_t i = 0; i < lo.rows(); ++i) {
    job_of_row(i);
  }
}

}  // namespace internal

template <typename SparseBlas>
class fsai0_preconditioner
    : public basic_preconditioner<fsai0_preconditioner<SparseBlas>, typename SparseBlas::scalar_type,
                                  typename SparseBlas::device_type, SparseBlas::compression> {
public:
  using base = basic_preconditioner<fsai0_preconditioner<SparseBlas>, typename SparseBlas::scalar_type,
                                    typename SparseBlas::device_type, SparseBlas::compression>;
  using Scalar = typename base::scalar_type;
  using Device = typename base::device_type;

  using vector_type = typename base::vector_type;
  using const_vector = typename base::const_vector;
  friend base;
  using const_sparse_view = sparse::basic_sparse_view<const Scalar, Device, sparse::sparse_format::csr>;
  using sparse_matrix = sparse::basic_sparse_matrix<Scalar, Device, sparse::sparse_format::csr>;
  // working on cpu.
  using const_cpu_view = sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr>;
  using cpu_view = sparse::basic_sparse_view<Scalar, device::cpu, sparse::sparse_format::csr>;
  using sparse_cpu_matrix = sparse::basic_sparse_matrix<Scalar, device::cpu, sparse::sparse_format::csr>;

  fsai0_preconditioner() = default;
  explicit fsai0_preconditioner(const const_sparse_view& view) : base(view) { this->compute({}); }

  fsai0_preconditioner(fsai0_preconditioner&&) = default;
  fsai0_preconditioner(const fsai0_preconditioner&) = delete;

  void factorize_impl() {
    auto matrix = this->matrix();
    auto n = matrix.rows(), nnz = matrix.nnz();
    auto cpu_matrix = sparse_cpu_matrix(n, n, nnz);
    auto orig_outer = cpu_matrix.outer_ptrs().view();
    auto orig_inner = cpu_matrix.inner_indices().view();
    auto orig_values = cpu_matrix.values().view();
    // copy the matrix to cpu.
    copy(orig_outer, matrix.outer_ptrs());
    copy(orig_inner, matrix.inner_indices());
    copy(orig_values, matrix.values());

    // create the lower triangular matrix.
    index_t lo_nnz = 0;
    for (index_t i = 0; i < n; ++i) {
      auto row_start = orig_outer[i];
      auto row_end = orig_outer[i + 1];
      for (index_t j = row_start; j < row_end; ++j) {
        if (orig_inner[j] <= i) {
          ++lo_nnz;
        }
      }
    }

    sparse_cpu_matrix lo(n, n, lo_nnz);
    auto lo_outer = lo.outer_ptrs().view();
    auto lo_inner = lo.inner_indices().view();
    {  // fill in the approx inverse of A in lo.
      index_t counter = 0;
      for (index_t i = 0; i < n; ++i) {
        lo_outer[i] = counter;
        auto row_start = orig_outer[i];
        auto row_end = orig_outer[i + 1];
        for (index_t j = row_start; j < row_end; ++j) {
          if (orig_inner[j] <= i) {
            lo_inner[counter] = orig_inner[j];
            ++counter;
          }
        }
      }
      lo_outer[n] = counter;
      MATHPRIM_ASSERT(counter == lo_nnz && "Counter is not equal to nnz.");
    }

    // fill in the values.
    sparse::basic_sparse_view<const Scalar, device::cpu, sparse::sparse_format::csr> a_input = cpu_matrix.const_view();
    sparse::basic_sparse_view<Scalar, device::cpu, sparse::sparse_format::csr> lo_input = lo.view();
    internal::fsai_compute<Scalar>(a_input, lo_input);

    // after that, copy the lo to device.
    lo_ = lo.to(Device());
    buffer_intern_ = make_buffer<Scalar, Device>(n);
    buffer_intern_.fill_bytes(0);
    bl_ = SparseBlas(lo_.const_view());
    has_compute_ = true;
  }

  const_sparse_view ainv() const noexcept { return lo_.const_view(); }

private:
  void apply_impl(vector_type y, const_vector x) {
    MATHPRIM_INTERNAL_CHECK_THROW(has_compute_, std::runtime_error, "The preconditioner has not been computed.");
    auto z = buffer_intern_.view();
    // z = lo * x.
    bl_.gemv(1, x, 0, z, false);
    // y = lo.T * y.
    bl_.gemv(1, z, 0, y, true);
  }

  bool has_compute_ = false;
  SparseBlas bl_;
  sparse_matrix lo_;                                            // the approx inverse decomposition of A.
  contiguous_buffer<Scalar, dshape<1>, Device> buffer_intern_;  // buffer for intermediate computation.
};
}  // namespace mathprim::sparse::iterative
