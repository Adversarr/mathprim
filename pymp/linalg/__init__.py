from typing import Any, Callable, Tuple, List, Union
import libpymp
import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor
import torch


def pcg(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    if callback:
        return libpymp.linalg.pcg_cb_no(A, b, x, rtol, max_iter, callback)
    else:
        return libpymp.linalg.pcg_no(A, b, x, rtol, max_iter, verbose)


def pcg_diagonal(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    if callback:
        return libpymp.linalg.pcg_cb_diagonal(A, b, x, rtol, max_iter, callback)
    else:
        return libpymp.linalg.pcg_diagonal(A, b, x, rtol, max_iter, verbose)

def pcg_ainv(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    if callback:
        return libpymp.linalg.pcg_cb_ainv(A, b, x, rtol, max_iter, callback)
    else:
        return libpymp.linalg.pcg_ainv(A, b, x, rtol, max_iter, verbose)


def pcg_with_ext_spai(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    ainv: csr_matrix,
    epsilon: float,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with External SPAI preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_with_ext_spai(A, b, x, ainv, epsilon, rtol, max_iter, verbose)


def pcg_with_ext_spai_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    ainv: csr_matrix,
    epsilon: float,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with External SPAI preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_with_ext_spai_cuda(A, b, x, ainv, epsilon, rtol, max_iter, verbose)

def pcg_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_no_cuda(A, b, x, rtol, max_iter, verbose)

def pcg_diagonal_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_diagonal_cuda(A, b, x, rtol, max_iter, verbose)

def pcg_ainv_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymp.linalg.pcg_ainv_cuda(A, b, x, rtol, max_iter, verbose)

def ainv(A: csr_matrix) -> csr_matrix:
    """
    Compute the content of the Approximated Inverse preconditioner.

    Returns
    -------
    csr_matrix
        The content of the Approximated Inverse preconditioner.
    """
    return libpymp.linalg.ainv_content(A)


def grid_laplacian_nd_dbc(
    grids: Union[List[int], np.ndarray],
    dtype=np.float32
) -> csr_matrix:
    """
    Construct the Laplacian operator on n-dimensional grid with Dirichlet boundary conditions.

    Returns
    -------
    csr_matrix
        The Laplacian operator.
    """

    if isinstance(grids, np.ndarray):
        grids = grids.tolist()

    if dtype == np.float32:
        return libpymp.linalg.grid_laplacian_nd_dbc_float32(grids)
    elif dtype == np.float64:
        return libpymp.linalg.grid_laplacian_nd_dbc_float64(grids)
    else:
        raise ValueError("dtype must be np.float32 or np.float64")

def cg_cuda_csr_direct(
    outer_ptrs: Tensor,
    inner_indices: Tensor,
    values: Tensor,
    rows: int,
    cols: int,
    b: Tensor,
    x: Tensor,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method on GPU.
    
    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    assert outer_ptrs.dtype == inner_indices.dtype == torch.int32
    assert values.dtype == b.dtype == x.dtype
    assert b.is_contiguous() and x.is_contiguous()
    return libpymp.linalg.pcg_no_cuda_direct(
        outer_ptrs=outer_ptrs,
        inner_indices=inner_indices,
        values=values,
        rows=rows,
        cols=cols,
        b=b,
        x=x,
        rtol=rtol,
        max_iter=max_iter,
        verbose=verbose,
    )

def pcg_cuda_csr_direct_diagonal(
    outer_ptrs: Tensor,
    inner_indices: Tensor,
    values: Tensor,
    rows: int,
    cols: int,
    b: Tensor,
    x: Tensor,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner on GPU.
    
    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    assert outer_ptrs.dtype == inner_indices.dtype == torch.int32
    assert values.dtype == b.dtype == x.dtype
    assert b.is_contiguous() and x.is_contiguous()
    return libpymp.linalg.pcg_diagonal_cuda_direct(
        outer_ptrs=outer_ptrs,
        inner_indices=inner_indices,
        values=values,
        rows=rows,
        cols=cols,
        b=b,
        x=x,
        rtol=rtol,
        max_iter=max_iter,
        verbose=verbose,
    )

def pcg_cuda_csr_direct_ic(
    outer_ptrs: Tensor,
    inner_indices: Tensor,
    values: Tensor,
    rows: int,
    cols: int,
    b: Tensor,
    x: Tensor,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Incomplete Cholesky preconditioner on GPU.
    
    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    assert outer_ptrs.dtype == inner_indices.dtype == torch.int32
    assert values.dtype == b.dtype == x.dtype
    assert b.is_contiguous() and x.is_contiguous()
    return libpymp.linalg.pcg_ic_cuda_direct(
        outer_ptrs=outer_ptrs,
        inner_indices=inner_indices,
        values=values,
        rows=rows,
        cols=cols,
        b=b,
        x=x,
        rtol=rtol,
        max_iter=max_iter,
        verbose=verbose,
   )

def pcg_cuda_csr_direct_ainv(
    outer_ptrs: Tensor,
    inner_indices: Tensor,
    values: Tensor,
    rows: int,
    cols: int,
    b: Tensor,
    x: Tensor,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner on GPU.
    
    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    assert outer_ptrs.dtype == inner_indices.dtype == torch.int32
    assert values.dtype == b.dtype == x.dtype
    assert b.is_contiguous() and x.is_contiguous()
    return libpymp.linalg.pcg_ainv_cuda_direct(
        outer_ptrs=outer_ptrs,
        inner_indices=inner_indices,
        values=values,
        rows=rows,
        cols=cols,
        b=b,
        x=x,
        rtol=rtol,
        max_iter=max_iter,
        verbose=verbose,
    )


def pcg_cuda_csr_direct_with_ext_spai(
    outer_ptrs: Tensor,
    inner_indices: Tensor,
    values: Tensor,
    rows: int,
    cols: int,
    ainv_outer_ptrs: Tensor,
    ainv_inner_indices: Tensor,
    ainv_values: Tensor,
    epsilon: float,
    b: Tensor,
    x: Tensor,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner on GPU.
    
    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    assert outer_ptrs.dtype == inner_indices.dtype == torch.int32
    assert values.dtype == b.dtype == x.dtype
    assert b.is_contiguous() and x.is_contiguous()
    return libpymp.linalg.pcg_with_ext_spai_cuda_direct(
        outer_ptrs=outer_ptrs,
        inner_indices=inner_indices,
        values=values,
        rows=rows,
        cols=cols,
        ainv_outer_ptrs=ainv_outer_ptrs,
        ainv_inner_indices=ainv_inner_indices,
        ainv_values=ainv_values,
        epsilon=epsilon,
        b=b,
        x=x,
        rtol=rtol,
        max_iter=max_iter,
        verbose=verbose,
    )


def ldlt(
    A: csr_matrix
) -> Any:
    """
    Compute the LDL^T decomposition of a symmetric positive definite matrix. (Eigen::SimplicialLDLT)

    Returns
    -------
    Solver
        The LDL^T decomposition.

    Examples
    --------
    >>> A = csr_matrix(...)
    >>> solver = ldlt(A)
    >>> solver.solve(b, x) # Solve Ax = b where x and b are vectors
    >>> solver.vsolve(b, x) # Solve A X = B where X and B are matrices
    """
    if A.dtype == np.float32:
        return libpymp.linalg.eigen_simplicial_ldlt_float32(A)
    elif A.dtype == np.float64:
        return libpymp.linalg.eigen_simplicial_ldlt_float64(A)

def llt(
    A: csr_matrix
) -> Any:
    """
    Compute the Cholesky decomposition of a symmetric positive definite matrix. (Eigen::SimplicialLLT)

    Returns
    -------
    Solver
        The Cholesky decomposition.

    Examples
    --------
    >>> A = csr_matrix(...)
    >>> solver = ldlt(A)
    >>> solver.solve(b, x) # Solve Ax = b where x and b are vectors
    >>> solver.vsolve(b, x) # Solve A X = B where X and B are matrices
    """
    if A.dtype == np.float32:
        return libpymp.linalg.eigen_simplicial_llt_float32(A)
    elif A.dtype == np.float64:
        return libpymp.linalg.eigen_simplicial_llt_float64(A)
