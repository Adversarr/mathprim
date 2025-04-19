from functools import partial
from scipy.sparse import csr_matrix
from typing import Callable, Optional, Tuple, Union, Literal
import numpy as np



def get_pcg_cuda(preconditioner: str) -> Callable:
    """
    Parameters
    ----------
    preconditioner : str

    Returns
    -------
    Callable
        CG Callable
    """
    from .cg_cuda import (
        pcg_cuda,
        pcg_diagonal_cuda,
        pcg_ainv_cuda,
        pcg_fsai_cuda,
        pcg_ic_cuda,
        pcg_with_ext_spai_cuda,
        pcg_with_ext_spai_cuda_scaled
    )

    preconditioners = {
        "none": pcg_cuda,
        "diagonal": pcg_diagonal_cuda,
        "ainv": pcg_ainv_cuda,
        "ic": pcg_ic_cuda,
        "ext_spai": pcg_with_ext_spai_cuda,
        "ext_spai_scaled": pcg_with_ext_spai_cuda_scaled,
        "fsai": pcg_fsai_cuda,
    }
    if preconditioner not in preconditioners:
        raise ValueError("Unknown preconditioner: {}".format(preconditioner))
    return preconditioners[preconditioner]


def get_pcg_host(preconditioner: str) -> Callable:
    """
    Parameters
    ----------
    preconditioner : str

    Returns
    -------
    Callable
        CG Callable
    """
    from .cg_host import (
        pcg,
        pcg_diagonal,
        pcg_ainv,
        pcg_fsai,
        pcg_ic,
        pcg_with_ext_spai,
        pcg_with_ext_spai_scaled,
    )

    preconditioners = {
        "none": pcg,
        "diagonal": pcg_diagonal,
        "ainv": pcg_ainv,
        "ic": pcg_ic,
        "ext_spai": pcg_with_ext_spai,
        "ext_spai_scaled": pcg_with_ext_spai_scaled,
        "fsai": pcg_fsai,
    }
    if preconditioner not in preconditioners:
        raise ValueError("Unknown preconditioner: {}".format(preconditioner))
    return preconditioners[preconditioner]


class PreconditionedConjugateGradient:
    def __init__(
        self,
        matrix: csr_matrix,
        device: Literal["cpu", "cuda"],
        preconditioner: Literal["none", "ainv", "ic", "diagonal", "ext_spai", "ext_spai_scaled"],
        dtype: Optional[np.dtype] = None,
    ):
        self.dtype = dtype or matrix.dtype
        if matrix.dtype != self.dtype:
            self.matrix: csr_matrix = matrix.astype(self.dtype)
        else:
            self.matrix = matrix

        self.device = device
        self.preconditioner: str = preconditioner

    def get_pcg(self):
        if self.device == "cuda":
            return get_pcg_cuda(self.preconditioner)
        elif self.device == "cpu":
            return get_pcg_host(self.preconditioner)
        else:
            raise ValueError("Unsupported device: {}".format(self.device))

    def __call__(
        self,
        b: np.ndarray,
        x: np.ndarray,
        rtol: float = 1e-4,
        max_iter: int = 0,
        verbose: int = 0,
        callback: Union[None, Callable] = None,
        ext_spai: Union[None, Tuple[csr_matrix, float]] = None,
    ) -> Tuple[int, float, float]:
        """
        Solve the linear system Ax = b using the conjugate gradient method on GPU.

        Returns
        -------
        int
            The number of iterations.
        float
            The time of precompute step.
        float
            The time taken to solve the linear system.
        """

        method = self.get_pcg()
        if self.preconditioner == "ext_spai" or self.preconditioner == "ext_spai_scaled":
            assert ext_spai is not None, "ext_spai must be provided for ext_spai preconditioner"
            ainv, eps = ext_spai
            method = partial(method, ainv=ainv, epsilon=eps)
        if self.device == "cpu" and self.preconditioner not in ["ext_spai", "ext_spai_scaled"]:
            # only cpu version support this.
            method = partial(method, callback=callback)
        return method(
            A=self.matrix,
            b=b,
            x=x,
            rtol=rtol,
            max_iter=max_iter,
            verbose=verbose,
        )
