from typing import Any
import libpymp
import numpy as np
from scipy.sparse import csr_matrix

def is_cholmod_available() -> bool:
    return libpymp.linalg.is_cholmod_available

def chol(
    A: csr_matrix
) -> Any:
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
    if A.dtype == np.float32:
        return libpymp.linalg.cholmod.cholmod_cholesky_float32(A)
    elif A.dtype == np.float64:
        return libpymp.linalg.cholmod.cholmod_cholesky_float64(A)
