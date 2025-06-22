
#include <iostream>
#include <mathprim/blas/cublas.cuh>
#include <mathprim/core/defines.hpp>
#include <mathprim/core/devices/cuda.cuh>
#include <mathprim/linalg/iterative/precond/diagonal.hpp>
#include <mathprim/linalg/iterative/precond/eigen_support.hpp>
#include <mathprim/linalg/iterative/precond/fsai0.hpp>
#include <mathprim/linalg/iterative/precond/ic_cusparse.hpp>
#include <mathprim/linalg/iterative/precond/sparse_inverse.hpp>
#include <mathprim/linalg/iterative/solver/cg.hpp>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/blas/cusparse.hpp>
#include <mathprim/supports/eigen_sparse.hpp>

#include "linalg.hpp"
#include "mathprim/linalg/iterative/precond/ainv.hpp"

using namespace mathprim;
using namespace helper;

template <typename Scalar>
using diagonal = sparse::iterative::diagonal_preconditioner<Scalar, device::cuda, sparse::sparse_format::csr,
                                                            blas::cublas<Scalar>>;

template <typename Scalar>
using ainv
    // = sparse::iterative::scaled_bridson_ainv_preconditioner<sparse::blas::cusparse<Scalar,
    // sparse::sparse_format::csr>>;
    = sparse::iterative::bridson_ainv_preconditioner<sparse::blas::cusparse<Scalar, sparse::sparse_format::csr>,
                                                     blas::cublas<Scalar>>;

template <typename Scalar>
using fsai = sparse::iterative::fsai0_preconditioner<sparse::blas::cusparse<Scalar, sparse::sparse_format::csr>>;

template <typename Scalar>
using ic = sparse::iterative::cusparse_ichol<Scalar, device::cuda, mathprim::sparse::sparse_format::csr>;

template <typename Scalar>
using no = sparse::iterative::none_preconditioner<Scalar, device::cuda, mathprim::sparse::sparse_format::csr>;

////////////////////////////////////////////////
/// CPU->GPU->CPU
////////////////////////////////////////////////
template <typename Flt, typename Precond, typename MatrixType>
std::tuple<index_t, double, double> solve_with_precond(
    const MatrixType& A,
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> b,
    nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> x,
    const Flt rtol,
    index_t max_iter,
    int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = SparseBlas;
  using Blas = blas::cublas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;
  using Sparse = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>;

  // Input validation
  // Validate input dimensions
  auto h_b_view = view(b.data(), make_shape(b.size()));
  auto h_x_view = view(x.data(), make_shape(x.size()));
  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Matrix A must be square.");
  }
  if (h_b_view.size() != A.rows() || h_x_view.size() != A.cols()) {
    throw std::invalid_argument(
      "Dimensions mismatch: b.size()=" + std::to_string(h_b_view.size()) +
      ", x.size()=" + std::to_string(h_x_view.size()) +
      ", A.rows()=" + std::to_string(A.rows()) +
      ", A.cols()=" + std::to_string(A.cols()));
  }

  // Set default max iterations if not specified
  if (max_iter == 0) {
    max_iter = A.rows();
  } else if (max_iter < 0) {
    throw std::invalid_argument("max_iter must be non-negative.");
  }

  // Setup GPU buffers
  auto d_b = make_cuda_buffer<Flt>(b.size());
  auto d_x = make_cuda_buffer<Flt>(x.size());
  auto b_view = d_b.view();
  auto x_view = d_x.view();

  // Copy data to GPU
  copy(b_view, h_b_view);
  copy(x_view, h_x_view);

  // Setup matrix on GPU
  auto view_A = eigen_support::view(A);
  auto d_A = Sparse(view_A.rows(), view_A.cols(), view_A.nnz());
  copy(d_A.outer_ptrs().view(), view_A.outer_ptrs());
  copy(d_A.inner_indices().view(), view_A.inner_indices());
  copy(d_A.values().view(), view_A.values());
  auto d_A_view = d_A.const_view();

  // Solve system
  // Time preconditioner setup
  auto start = time_now();
  Solver solver(d_A_view);
  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  sparse::convergence_result<Flt> result;
  auto prec_setup = time_elapsed(start);
  
  // Time solver iterations
  start = time_now();

  if (verbose > 0) {
    result = solver.solve(x_view, b_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.solve(x_view, b_view, criteria);
  }
  auto solve = time_elapsed(start);

  // Copy result back to CPU
  copy(h_x_view, x_view);
  return {result.iterations_, prec_setup, solve};
}

template <typename Flt, typename Precond>
std::tuple<index_t, double, double> cg_cuda(Eigen::SparseMatrix<Flt, Eigen::RowMajor> A,
                                           nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> b,
                                           nb::ndarray<Flt, nb::shape<-1>, nb::device::cpu> x,
                                           const Flt rtol,
                                           index_t max_iter,
                                           int verbose) {
  return solve_with_precond<Flt, Precond>(A, b, x, rtol, max_iter, verbose);
}

////////////////////////////////////////////////
/// CUDA direct
////////////////////////////////////////////////
template <typename Flt, typename Precond>
static std::tuple<index_t, double, double> cg_cuda_csr_direct(          //
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
  using LinearOp = SparseBlas;
  using Blas = blas::cublas<Flt>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;
  using SpView = sparse::basic_sparse_view<const Flt, device::cuda, mathprim::sparse::sparse_format::csr>;

  auto b_view = view<device::cuda>(b.data(), make_shape(b.size())).as_const();
  auto x_view = view<device::cuda>(x.data(), make_shape(x.size()));
  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b_view.size() != rows || x_view.size() != cols) {
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
  SpView matrix_v(p_values, p_outer, p_inner, rows, cols, nnz, sparse::sparse_property::symmetric);

  auto start = time_now();
  Solver solver(matrix_v);
  auto prec_setup = time_elapsed(start);
  start = time_now();

  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  sparse::convergence_result<Flt> result;
  if (verbose > 0) {
    result = solver.solve(x_view, b_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.solve(x_view, b_view, criteria);
  }
  auto solve = time_elapsed(start);

  return {result.iterations_, prec_setup, solve};
}

template <typename Flt = float>
static std::tuple<index_t, double, double> pcg_with_ext_spai(  //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,        //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,          //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,          //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,     //
    Flt epsilon,                                               //
    const Flt& rtol,                                           //
    index_t max_iter,                                          //
    int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = SparseBlas;
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

  // auto ainv_host = eigen_support::view(ainv);
  // auto ainv_device = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>(
  //     ainv_host.rows(), ainv_host.cols(), ainv_host.nnz());
  // auto view_ainv = ainv_device.view();
  // copy(view_ainv.outer_ptrs(), ainv_host.outer_ptrs());
  // copy(view_ainv.inner_indices(), ainv_host.inner_indices());
  // copy(view_ainv.values(), ainv_host.values());

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
  auto start = time_now();
  Solver solver(view_device.as_const());
  // solver.preconditioner().derived().set_approximation(view_ainv.as_const(), epsilon);
  solver.preconditioner().derived().set_approximation(ainv, epsilon);
  auto prec_setup = time_elapsed(start);
  start = time_now();
  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  sparse::convergence_result<Flt> result;

  if (verbose > 0) {
    result = solver.solve(d_x, d_b, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.solve(d_x, d_b, criteria);
  }
  auto solve = time_elapsed(start);

  copy(h_x, d_x);
  return {result.iterations_, prec_setup, solve};
}

template <typename Flt = float>
static std::tuple<index_t, double, double> pcg_with_ext_spai_scaled(  //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& A,               //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> b,                 //
    nb::ndarray<Flt, nb::ndim<1>, nb::device::cpu> x,                 //
    const Eigen::SparseMatrix<Flt, Eigen::RowMajor>& ainv,            //
    Flt epsilon,                                                      //
    bool already_scaled,                                              //
    const Flt& rtol,                                                  //
    index_t max_iter,                                                 //
    int verbose) {
  using SparseBlas = mp::sparse::blas::cusparse<Flt, sparse::sparse_format::csr>;
  using LinearOp = SparseBlas;
  using Blas = mp::blas::cublas<Flt>;
  using Precond = mp::sparse::iterative::scale_sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;

  // 1. Setup Solver & Preconditioner.
  auto matrix_host = eigen_support::view(A);
  auto matrix_device = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>(
      matrix_host.rows(), matrix_host.cols(), matrix_host.nnz());
  auto view_device = matrix_device.view();
  copy(view_device.outer_ptrs(), matrix_host.outer_ptrs());
  copy(view_device.inner_indices(), matrix_host.inner_indices());
  copy(view_device.values(), matrix_host.values());

  // auto ainv_host = eigen_support::view(ainv);
  // auto ainv_device = sparse::basic_sparse_matrix<Flt, device::cuda, mathprim::sparse::sparse_format::csr>(
  //     ainv_host.rows(), ainv_host.cols(), ainv_host.nnz());
  // auto view_ainv = ainv_device.view();
  // copy(view_ainv.outer_ptrs(), ainv_host.outer_ptrs());
  // copy(view_ainv.inner_indices(), ainv_host.inner_indices());
  // copy(view_ainv.values(), ainv_host.values());

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
  auto start = time_now();
  Solver solver(view_device.as_const());
  // solver.preconditioner().derived().set_approximation(view_ainv.as_const(), epsilon);
  solver.preconditioner().derived().set_approximation(ainv, epsilon, already_scaled);
  auto prec = time_elapsed(start);
  start = time_now();
  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  sparse::convergence_result<Flt> result;

  if (verbose > 0) {
    result = solver.solve(d_x, d_b, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.solve(d_x, d_b, criteria);
  }
  auto solve = time_elapsed(start);

  copy(h_x, d_x);
  return {result.iterations_, prec, solve};
}

template <typename Flt>
static std::tuple<index_t, double, double> pcg_with_ext_spai_cuda_direct(    //
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
  using LinearOp = SparseBlas;
  using Blas = blas::cublas<Flt>;
  using Precond = mp::sparse::iterative::sparse_preconditioner<SparseBlas, Blas>;
  using Solver = mp::sparse::iterative::cg<Flt, mp::device::cuda, LinearOp, Blas, Precond>;
  using SpView = sparse::basic_sparse_view<const Flt, device::cuda, mathprim::sparse::sparse_format::csr>;

  auto b_view = view<device::cuda>(b.data(), make_shape(b.size())).as_const();
  auto x_view = view<device::cuda>(x.data(), make_shape(x.size()));

  if (b.ndim() != 1 || x.ndim() != 1) {
    throw std::invalid_argument("b and x must be 1D arrays.");
  }

  if (b_view.size() != rows || x_view.size() != cols) {
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
  auto start = time_now();
  Solver solver(view_a);
  solver.preconditioner().derived().set_approximation(view_ainv, eps);
  auto prec = time_elapsed(start);
  start = time_now();

  // 2. Setup working vectors.

  sparse::convergence_criteria<Flt> criteria{max_iter, rtol};
  sparse::convergence_result<Flt> result;
  if (verbose > 0) {
    result = solver.solve(x_view, b_view, criteria, [verbose](index_t iter, Flt norm) {
      if (iter % verbose == 0) {
        std::cout << "Iteration: " << iter << ", Norm: " << norm << std::endl;
      }
    });
  } else {
    result = solver.solve(x_view, b_view, criteria);
  }
  auto solve = time_elapsed(start);
  return {result.iterations_, prec, solve};
}

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
  BIND_TRANSFERING_GPU_TYPE(flt, fsai);     \
  BIND_DIRECT(flt, no);                     \
  BIND_DIRECT(flt, diagonal);               \
  BIND_DIRECT(flt, ainv);                   \
  BIND_DIRECT(flt, ic);                     \
  BIND_DIRECT(flt, fsai);

template <typename Flt>
static void bind_extra(nb::module_& m) {
  m.def("pcg_with_ext_spai_cuda", &pcg_with_ext_spai<Flt>,
        "Preconditioned CG on GPU (cpu->gpu->cpu) (with SPAI precond.)",
        nb::arg("A").noconvert(),                         // System to solve
        nb::arg("b").noconvert(),                         // Right-hand side
        nb::arg("x").noconvert(),                         // Initial guess
        nb::arg("ainv").noconvert(), nb::arg("epsilon"),  // Approximate inverse
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);

  m.def("pcg_with_ext_spai_cuda_scaled", &pcg_with_ext_spai_scaled<Flt>,
        "Preconditioned CG on GPU (cpu->gpu->cpu) (with SPAI precond.)",
        nb::arg("A").noconvert(),                                                    // System to solve
        nb::arg("b").noconvert(),                                                    // Right-hand side
        nb::arg("x").noconvert(),                                                    // Initial guess
        nb::arg("ainv").noconvert(), nb::arg("epsilon"), nb::arg("already_scaled"),  // Approximate inverse
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);

  m.def("pcg_with_ext_spai_cuda_direct", &pcg_with_ext_spai_cuda_direct<Flt>,                                    //
        "Preconditioned CG on GPU (direct) (with SPAI precond.)",                                                //
        nb::arg("outer_ptrs").noconvert(), nb::arg("inner_indices").noconvert(), nb::arg("values").noconvert(),  //
        nb::arg("rows"), nb::arg("cols"),                                                                        //
        nb::arg("ainv_outer_ptrs").noconvert(), nb::arg("ainv_inner_indices").noconvert(),
        nb::arg("ainv_values").noconvert(),                  //
        nb::arg("epsilon"),                                  //
        nb::arg("b").noconvert(), nb::arg("x").noconvert(),  //
        nb::arg("rtol") = 1e-4f, nb::arg("max_iter") = 0, nb::arg("verbose") = 0);
}

void bind_linalg_cuda(nb::module_& m) {
  BIND_ALL(float);
  BIND_ALL(double);
  bind_extra<float>(m);
  bind_extra<double>(m);
}
