from typing import Callable, Tuple, List, Union
import libpymathprim
import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "ainv",
    "pcg",
    "pcg_diagonal",
    "pcg_ainv",
    "pcg_ic",
    "pcg_with_ext_spai",
    "grid_laplacian_nd_dbc",
]


def pcg(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method.

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
        return libpymathprim.linalg.pcg_cb_no(A, b, x, rtol, max_iter, callback)
    else:
        return libpymathprim.linalg.pcg_no(A, b, x, rtol, max_iter, verbose)


def pcg_diagonal(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with diagonal preconditioner.

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
        return libpymathprim.linalg.pcg_cb_diagonal(A, b, x, rtol, max_iter, callback)
    else:
        return libpymathprim.linalg.pcg_diagonal(A, b, x, rtol, max_iter, verbose)


def pcg_ainv(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Approximated Inverse preconditioner.

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
        return libpymathprim.linalg.pcg_cb_ainv(A, b, x, rtol, max_iter, callback)
    else:
        return libpymathprim.linalg.pcg_ainv(A, b, x, rtol, max_iter, verbose)


def pcg_ic(
    A: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
    rtol: float = 1e-4,
    max_iter: int = 0,
    verbose: int = 0,
    callback: Union[None, Callable] = None,
) -> Tuple[int, float, float]:
    """
    Solve the linear system Ax = b using the conjugate gradient method with Incomplete Cholesky preconditioner.

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
        return libpymathprim.linalg.pcg_cb_ic(A, b, x, rtol, max_iter, callback)
    else:
        return libpymathprim.linalg.pcg_ic(A, b, x, rtol, max_iter, verbose)


def pcg_with_ext_spai(
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
    return libpymathprim.linalg.pcg_with_ext_spai(
        A, b, x, ainv, epsilon, rtol, max_iter, verbose
    )


def ainv(A: csr_matrix) -> csr_matrix:
    """
    Compute the content of the Approximated Inverse preconditioner.

    Returns
    -------
    csr_matrix
        The content of the Approximated Inverse preconditioner.
    """
    return libpymathprim.linalg.ainv_content(A)


def grid_laplacian_nd_dbc(
    grids: Union[List[int], np.ndarray], dtype=np.float32
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
        return libpymathprim.linalg.grid_laplacian_nd_dbc_float32(grids)
    elif dtype == np.float64:
        return libpymathprim.linalg.grid_laplacian_nd_dbc_float64(grids)
    else:
        raise ValueError("dtype must be np.float32 or np.float64")
