from typing import Union
import libpymathprim
import numpy as np
from scipy.sparse import csr_matrix


def laplacian(
    vert: np.ndarray,
    edge: np.ndarray,
    edge_weights: Union[np.ndarray, None] = None,
) -> csr_matrix:
    """
    Construct the Laplacian matrix of a mesh.
    """
    if edge_weights is not None:  # If edge weights are provided, use them
        return libpymathprim.geometry.weighted_laplacian(vert, edge, edge_weights)
    else:
        return libpymathprim.geometry.laplacian(vert, edge)


def lumped_mass(vert: np.ndarray, edge: np.ndarray) -> csr_matrix:
    """
    Construct the lumped mass matrix of a mesh.
    """
    return libpymathprim.geometry.lumped_mass(vert, edge)
