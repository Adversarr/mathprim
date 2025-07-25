from typing import Any, Callable, Tuple, List, Union
import libpymathprim
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import Tensor

__all__ = [
    "pcg_with_ext_spai_cuda",
    "pcg_cuda",
    "pcg_diagonal_cuda",
    "pcg_ainv_cuda",
    "pcg_ic_cuda",
    "pcg_with_ext_cholesky",
    "cg_cuda_csr_direct",
    "pcg_cuda_csr_direct_diagonal",
    "pcg_cuda_csr_direct_ic",
    "pcg_cuda_csr_direct_ainv",
    "pcg_cuda_csr_direct_with_ext_spai",
]


def pcg_with_ext_spai_cuda_scaled(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    ainv: csr_matrix,
    epsilon: float,
    already_scaled: bool = True,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with External SPAI preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymathprim.linalg.pcg_with_ext_spai_cuda_scaled(
        A, b, x, ainv, epsilon, already_scaled, rtol, max_iter, verbose
    )


def pcg_with_ext_spai_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    ainv: csr_matrix,
    epsilon: float,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with External SPAI preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymathprim.linalg.pcg_with_ext_spai_cuda(
        A, b, x, ainv, epsilon, rtol, max_iter, verbose
    )


def pcg_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymathprim.linalg.pcg_no_cuda(A, b, x, rtol, max_iter, verbose)


def pcg_diagonal_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymathprim.linalg.pcg_diagonal_cuda(A, b, x, rtol, max_iter, verbose)


def pcg_ainv_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner on GPU.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymathprim.linalg.pcg_ainv_cuda(A, b, x, rtol, max_iter, verbose)

def pcg_fsai_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with FSAI0 preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to prefactorize the system.
    float
        The time taken to solve the linear system.
    """
    if callback:
        return libpymathprim.linalg.pcg_cb_fsai_cuda(A, b, x, rtol, max_iter, callback)
    else:
        return libpymathprim.linalg.pcg_fsai_cuda(A, b, x, rtol, max_iter, verbose)

def pcg_ic_cuda(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Incomplete Cholesky preconditioner.

    Returns
    -------
    int
        The number of iterations.
    float
        The time taken to solve the linear system.
    """
    return libpymathprim.linalg.pcg_ic_cuda(A, b, x, rtol, max_iter, verbose)


def pcg_with_ext_cholesky(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    lower_l: csr_matrix,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with external incomplete Cholesky preconditioner.
    
    This function uses a pre-computed lower triangular matrix from an incomplete Cholesky
    factorization as the preconditioner for the conjugate gradient solver.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The coefficient matrix of the linear system.
    b : numpy.ndarray
        The right-hand side vector of the linear system.
    x : numpy.ndarray
        The initial guess for the solution vector.
    lower_l : scipy.sparse.csr_matrix
        The lower triangular matrix from an incomplete Cholesky factorization to be used as preconditioner.
    rtol : float, optional
        The relative tolerance for the stopping criterion. Default is 1e-4.
    max_iter : int, optional
        The maximum number of iterations. If 0, it will be set to the number of rows in A. Default is 0.
    verbose : int, optional
        The verbosity level. If greater than 0, progress information will be printed. Default is 0.

    Returns
    -------
    int
        The number of iterations performed.
    float
        The time taken to setup the preconditioner (in seconds).
    float
        The time taken to solve the linear system (in seconds).
    """
    return libpymathprim.linalg.pcg_with_ext_cholesky(A, b, x, lower_l, rtol, max_iter, verbose)


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
) -> Tuple[int, float, float]:
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
    return libpymathprim.linalg.pcg_no_cuda_direct(
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
) -> Tuple[int, float, float]:
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
    return libpymathprim.linalg.pcg_diagonal_cuda_direct(
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
) -> Tuple[int, float, float]:
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
    return libpymathprim.linalg.pcg_ic_cuda_direct(
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
) -> Tuple[int, float, float]:
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
    return libpymathprim.linalg.pcg_ainv_cuda_direct(
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
) -> Tuple[int, float, float]:
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
    return libpymathprim.linalg.pcg_with_ext_spai_cuda_direct(
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

