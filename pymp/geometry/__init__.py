from typing import Union
import libpymp
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
    if edge_weights:  # If edge weights are provided, use them
        return libpymp.geometry.laplacian(vert, edge, edge_weights)
    else:
        return libpymp.geometry.laplacian(vert, edge)


def lumped_mass(vert: np.ndarray, edge: np.ndarray) -> csr_matrix:
    """
    Construct the lumped mass matrix of a mesh.
    """
    return libpymp.geometry.lumped_mass(vert, edge)
