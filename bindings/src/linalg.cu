#include <iostream>
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/defines.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/linalg/iterative/precond/approx_inv.hpp>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>
#include <mathprim/linalg/iterative/precond/ic_cusparse.hpp>
#include <mathprim/linalg/iterative/precond/sparse_inverse.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/blas/cusparse.hpp>
#include <mathprim/supports/eigen_sparse.hpp>

#include "linalg.hpp"

using namespace mathprim;

////////////////////////////////////////////////
/// CPU->GPU->CPU
////////////////////////////////////////////////
template <typename Flt, typename Precond = sparse::iterative::none_preconditioner<Flt, device::cuda>>
std::pair<index_t, double> cg_cuda(const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,  //
                                   nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> b,  //
                                   nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> x,  //
                                   const Flt rtol,                                      //
                                   index_t max_iter,                                    //
                                   int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = blas::cublas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b.size() != A.rows() || x.size() != A.cols()) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  auto h_b_view = view(b.data(), make_shape(b.size()));
  auto h_x_view = view(x.data(), make_shape(x.size()));
  if (max_iter == 0) {
    max_iter = A.rows();
  }
  auto d_b = make_cuda_buffer<Flt>(b.size());
  auto d_x = make_cuda_buffer<Flt>(x.size());
  auto b_view = d_b.view();
  auto x_view = d_x.view();
  auto view_A = eigen_support::view(A);

  auto d_A = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>(
      view_A.rows(), view_A.cols(), view_A.nnz());

  copy(b_view, h_b_view);
  copy(x_view, h_x_view);

  copy(d_A.outer_ptrs().view(), view_A.outer_ptrs());
  copy(d_A.inner_indices().view(), view_A.inner_indices());
  copy(d_A.values().view(), view_A.values());

  auto d_A_view = d_A.const_view();
  Solver solver(LinearOp(d_A_view), Blas{}, Precond{d_A_view});
  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.apply(b_view, x_view, criteria);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  copy(h_x_view, x_view);
  return std::make_pair(result.iterations_, seconds);
}

////////////////////////////////////////////////
/// CUDA direct
////////////////////////////////////////////////
template <typename Flt, typename Precond>
static std::pair<index_t, double> cg_cuda_csr_direct(                   //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> outer_ptrs,     //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> inner_indices,  //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cuda> values,             //
    index_t rows, index_t cols,                                         //
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cuda> b,                //
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cuda> x,                //
    const Flt rtol,                                                     //
    index_t max_iter,                                                   //
    int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = blas::cublas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;
  using SpView = sparse::basic_sparse_view<const Flt, device::cuda, mathprim::sparse::sparse_format::csr>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b.size() != rows || x.size() != cols) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  if (max_iter == 0) {
    max_iter = rows;
  }

  const Flt* p_values = values.data();
  const index_t* p_outer = outer_ptrs.data();
  const index_t* p_inner = inner_indices.data();
  const index_t nnz = static_cast<index_t>(values.size());
  if (static_cast<index_t>(outer_ptrs.size()) != rows + 1) {
    throw std::invalid_argument("Invalid outer_ptrs size.");
  }
  if (static_cast<index_t>(inner_indices.size()) != nnz) {
    throw std::invalid_argument("Invalid inner_indices size.");
  }

  SpView view_A(p_values, p_outer, p_inner, rows, cols, nnz, sparse::sparse_property::symmetric);

  Solver solver(LinearOp(view_A), Blas{}, Precond{view_A});

  auto b_view = view<device::cuda>(b.data(), make_shape(b.size())).as_const();
  auto x_view = view<device::cuda>(x.data(), make_shape(x.size()));

  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.apply(b_view, x_view, criteria);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

template <typename Flt = float>
static std::pair<index_t, double> pcg_with_ext_spai(        //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,     //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,       //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,       //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,  //
    Flt epsilon,                                            //
    const Flt& rtol,                                        //
    index_t max_iter,                                       //
    int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = mp::blas::cublas<Flt>;
  using Precond = mp::sparse::iterative::sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;

  // 1. Setup Solver & Preconditioner.
  auto matrix_host = eigen_support::view(A);
  auto matrix_device = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>(
      matrix_host.rows(), matrix_host.cols(), matrix_host.nnz());
  auto view_device = matrix_device.view();
  copy(view_device.outer_ptrs(), matrix_host.outer_ptrs());
  copy(view_device.inner_indices(), matrix_host.inner_indices());
  copy(view_device.values(), matrix_host.values());

  auto ainv_host = eigen_support::view(ainv);
  auto ainv_device = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>(
      ainv_host.rows(), ainv_host.cols(), ainv_host.nnz());
  auto view_ainv = ainv_device.view();
  copy(view_ainv.outer_ptrs(), ainv_host.outer_ptrs());
  copy(view_ainv.inner_indices(), ainv_host.inner_indices());
  copy(view_ainv.values(), ainv_host.values());

  Solver solver(LinearOp{view_device.as_const()}, Blas{}, Precond(view_ainv.as_const(), epsilon));

  // 2. Prepare the buffers.
  auto h_b = view(b.data(), make_shape(b.size()));
  auto h_x = view(x.data(), make_shape(x.size()));
  auto d_b_buf = make_cuda_buffer<Flt>(b.size());
  auto d_x_buf = make_cuda_buffer<Flt>(x.size());
  auto d_b = d_b_buf.view();
  auto d_x = d_x_buf.view();
  copy(d_b, h_b);
  copy(d_x, h_x);

  // 3. Solve the system.
  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;

  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(d_b, d_x, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.apply(d_b, d_x, criteria);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();

  copy(h_x, d_x);
  return std::make_pair(result.iterations_, seconds);
}

template <typename Flt>
static std::pair<index_t, double> cg_direct_with_ext_spai(                   //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> outer_ptrs,          //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> inner_indices,       //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cuda> values,                  //
    index_t rows, index_t cols,                                              //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> ainv_outer_ptrs,     //
    nb::ndarray<index_t, nb::ndim<1>, nb::device::cuda> ainv_inner_indices,  //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cuda> ainv_values,             //
    Flt eps,                                                                 //
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cuda> b,                     //
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cuda> x,                     //
    const Flt rtol,                                                          //
    index_t max_iter,                                                        //
    int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = sparse::iterative::sparse_matrix<SparseBlas>;
  using Blas = blas::cublas<Flt>;
  using Precond = mp::sparse::iterative::sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;
  using SpView = sparse::basic_sparse_view<const Flt, device::cuda, mathprim::sparse::sparse_format::csr>;

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b.size() != rows || x.size() != cols) {
    throw std::invalid_argument("b and x must have the same size as the matrix.");
  }

  if (max_iter == 0) {
    max_iter = rows;
  }
  MATHPRIM_INTERNAL_CHECK_THROW(max_iter > 0, std::runtime_error, "max_iter must be positive.");

  // 1. Setup Solver & Preconditioner.
  const Flt *p_values = values.data(), *p_ainv_values = ainv_values.data();
  const index_t *p_outer = outer_ptrs.data(), *p_inner = inner_indices.data();
  const index_t *p_ainv_outer = ainv_outer_ptrs.data(), *p_ainv_inner = ainv_inner_indices.data();
  const index_t nnz = static_cast<index_t>(values.size()), ainv_nnz = static_cast<index_t>(ainv_values.size());
  if (static_cast<index_t>(outer_ptrs.size()) != rows + 1) {
    throw std::invalid_argument("Invalid outer_ptrs size.");
  }
  if (static_cast<index_t>(inner_indices.size()) != nnz) {
    throw std::invalid_argument("Invalid inner_indices size.");
  }

  SpView view_a(p_values, p_outer, p_inner, rows, cols, nnz, sparse::sparse_property::symmetric);
  SpView view_ainv(p_ainv_values, p_ainv_outer, p_ainv_inner, rows, cols, ainv_nnz, sparse::sparse_property::general);
  Solver solver(LinearOp(view_a), Blas{}, Precond{view_ainv, eps});

  // 2. Setup working vectors.
  auto b_view = view<device::cuda>(b.data(), make_shape(b.size())).as_const();
  auto x_view = view<device::cuda>(x.data(), make_shape(x.size()));

  sparse::iterative::convergence_criteria<Flt> criteria{
    .max_iterations_ = max_iter,
    .norm_tol_ = rtol,
  };
  sparse::iterative::convergence_result<Flt> result;
  auto start = std::chrono::high_resolution_clock::now();
  if (verbose > 0) {
    result = solver.apply(b_view, x_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.apply(b_view, x_view, criteria);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double seconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(duration).count();
  return std::make_pair(result.iterations_, seconds);
}

template <typename Scalar>
using diagonal = sparse::iterative::diagonal_preconditioner<Scalar, device::cuda, sparse::sparse_format::csr,
                                                            blas::cublas<Scalar>>;

template <typename Scalar>
using ainv
    = sparse::iterative::approx_inverse_preconditioner<sparse::blas::cusparse<Scalar, sparse::sparse_format::csr>>;

template <typename Scalar>
using ic = sparse::iterative::cusparse_ichol<Scalar, device::cuda, mathprim::sparse::sparse_format::csr>;

template <typename Scalar>
using no = sparse::iterative::none_preconditioner<Scalar, device::cuda>;

#define BIND_TRANSFERING_GPU_TYPE(flt, preconditioning)                                                            \
  m.def(TOSTR(pcg_##preconditioning##_cuda), &cg_cuda<flt, preconditioning<flt>>,                                  \
        "Preconditioned CG on GPU (cpu->gpu->cpu) (with " #preconditioning " precond.)", nb::arg("A").noconvert(), \
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,      \
        nb::arg("verbose") = 0)

#define BIND_DIRECT(flt, preconditioning)                                                                            \
  m.def(TOSTR(pcg_##preconditioning##_cuda_direct), &cg_cuda_csr_direct<flt, preconditioning<flt>>,                  \
        "Preconditioned CG on GPU (direct) (with " #preconditioning " precond.)", nb::arg("outer_ptrs").noconvert(), \
        nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(), nb::arg("rows"), nb::arg("cols"),       \
        nb::arg("b").noconvert(), nb::arg("x").noconvert(), nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0,        \
        nb::arg("verbose") = 0)

#define BIND_ALL(flt)                       \
  BIND_TRANSFERING_GPU_TYPE(flt, no);       \
  BIND_TRANSFERING_GPU_TYPE(flt, diagonal); \
  BIND_TRANSFERING_GPU_TYPE(flt, ainv);     \
  BIND_TRANSFERING_GPU_TYPE(flt, ic);       \
  BIND_DIRECT(flt, no);                     \
  BIND_DIRECT(flt, diagonal);               \
  BIND_DIRECT(flt, ainv);                   \
  BIND_DIRECT(flt, ic)

template <typename Flt>
static void bind_extra(nb::module_& m) {
  m.def("pcg_with_ext_spai_cuda", &pcg_with_ext_spai<Flt>,
        "Preconditioned CG on GPU (cpu->gpu->cpu) (with SPAI precond.)",
        nb::arg("A").noconvert(),                         // System to solve
        nb::arg("b").noconvert(),                         // Right-hand side
        nb::arg("x").noconvert(),                         // Initial guess
        nb::arg("ainv").noconvert(), nb::arg("epsilon"),  // Approximate inverse
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);

  m.def("pcg_with_ext_spai_cuda_direct", &cg_direct_with_ext_spai<Flt>,                                          //
        "Preconditioned CG on GPU (direct) (with SPAI precond.)",                                                //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("ainv_outer_ptrs").noconvert(), nb::arg("ainv_inner_indices").noconvert(),
        nb::arg("ainv_values").noconvert(),                  //
        nb::arg("epsilon"),                                  //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),  //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
}

// static std::pair<index_t, double> pcg_with_ext_spai(        //
//   const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,     //
//   nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,       //
//   nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,       //
//   const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,  //
//   Flt epsilon,                                            //
//   const Flt& rtol,                                        //
//   index_t max_iter,                                       //
//   int verbose) {

void bind_linalg_cuda(nb::module_& m) {
  BIND_ALL(float);
  BIND_ALL(double);
  bind_extra<float>(m);
  bind_extra<double>(m);
}
