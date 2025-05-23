from typing import Any
import libpymathprim
import numpy as np
from scipy.sparse import csr_matrix


def is_cholmod_available() -> bool:
    return libpymathprim.linalg.is_cholmod_available


def chol(A: csr_matrix) -> Any:
    """
    Compute the Cholesky decomposition of a sparse matrix A. (CHOLMOD)

    Returns
    -------
    Solver
        CHOLMOD solver.

    Examples
    --------
    >>> A = csr_matrix(...)
    >>> solver = chol(A)
    >>> solver.solve(b, x) # Solve Ax = b where x and b are vectors
    >>> solver.vsolve(b, x) # Solve A X = B where X and B are matrices
    """
    assert is_cholmod_available(), "CHOLMOD is not available. Please install it to use this function."
    if A.dtype == np.float32:
        return libpymathprim.linalg.cholmod.cholmod_cholesky_float32(A)
    elif A.dtype == np.float64:
        return libpymathprim.linalg.cholmod.cholmod_cholesky_float64(A)


def ldlt(A: csr_matrix) -> Any:
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
        return libpymathprim.linalg.eigen_simplicial_ldlt_float32(A)
    elif A.dtype == np.float64:
        return libpymathprim.linalg.eigen_simplicial_ldlt_float64(A)


def llt(A: csr_matrix) -> Any:
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
        return libpymathprim.linalg.eigen_simplicial_llt_float32(A)
    elif A.dtype == np.float64:
        return libpymathprim.linalg.eigen_simplicial_llt_float64(A)
