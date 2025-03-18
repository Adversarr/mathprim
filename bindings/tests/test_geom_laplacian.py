# Triangle mesh Laplacian matrix tests
import numpy as np
import matplotlib.pyplot as plt
from pymp.geometry import laplacian

def mesh(rows, cols):
    # Create a simple 2D mesh
    n = rows * cols
    vertices = np.zeros((n, 2))
    faces = []
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            vertices[k, 0] = j
            vertices[k, 1] = i
            if i < rows - 1 and j < cols - 1:
                faces.append([k, k + 1, k + cols])
                faces.append([k + 1, k + cols + 1, k + cols])
    return vertices.astype(np.float32), np.array(faces).astype(np.int32)

vert, face = mesh(3, 5)

plt.figure()
plt.triplot(vert[:, 0], vert[:, 1], face)
plt.plot(vert[:, 0], vert[:, 1], 'o')

L = laplacian(vert, face)

print(L.todense())
plt.show()