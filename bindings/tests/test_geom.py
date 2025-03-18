from pymp.geometry import laplacian, lumped_mass
import numpy as np

vert = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
tri = np.array([[0, 1, 2]], dtype=np.int32)

lap = laplacian(vert, tri).todense()
print(lap, np.linalg.det(lap))
print(lumped_mass(vert, tri).todense())
