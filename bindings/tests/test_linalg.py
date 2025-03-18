from pymp.linalg import pcg, pcg_cuda, pcg_diagonal, pcg_ainv, pcg_diagonal_cuda, pcg_ainv_cuda, ainv, grid_laplacian_nd_dbc, pcg_with_ext_spai, pcg_with_ext_spai_cuda

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from timeit import Timer

x = np.array([1,2,3], dtype=np.float32)
b = np.array([3,2,1], dtype=np.float32)
A = csr_matrix(
  [[3, 1, 0],
   [1, 3, 1],
   [0, 1, 3]], dtype=np.float32)


print(pcg(A, b, x, 1e-4, 20, 1))
print(x)
print(A @ x - b)
x = np.array([1,2,3], dtype=np.float32)
def callback(iter: int, res: float):
    print(iter, res, x)
pcg(A, b, x, 1e-4, 20, callback=callback)

x = np.array([1,2,3], dtype=np.float32)
print(pcg_cuda(A, b, x, 1e-4, 20, 1))
print(x)
print(A @ x - b)

for method in [pcg, pcg_cuda, pcg_diagonal, pcg_ainv, pcg_diagonal_cuda, pcg_ainv_cuda]:
    print(method.__name__)
    x = np.array([1,2,3], dtype=np.float32)
    print(method(A, b, x, 1e-4, 20, 1))
    print(x)
    print(A @ x - b)

print(ainv(A))
print("=== pcg_with_ext_spai ===")
x = np.array([1,2,3], dtype=np.float32)
print(pcg_with_ext_spai(A, b, x, ainv(A), 1e-6, 1e-4, 20, 1))
print("=== pcg_with_ext_spai_cuda ===")
x = np.array([1,2,3], dtype=np.float32)
print(pcg_with_ext_spai_cuda(A, b, x, ainv(A), 1e-6, 1e-4, 20, 1))

def laplace_2d(n):
    return grid_laplacian_nd_dbc([n, n], dtype=np.float32)

n = 128
cnt = 3
A = laplace_2d(n)

ai = ainv(A)


def eval_once(solver):
    x = np.ones(n*n, dtype=np.float32)
    b = A @ x
    x = np.zeros(n*n, dtype=np.float32)
    it, er = solver(A, b, x, 1e-6, n * n * 4, 0)
    print(it, er)

for method in [pcg, pcg_cuda, pcg_diagonal, pcg_ainv, pcg_diagonal_cuda, pcg_ainv_cuda]:
    print(method.__name__)
    t = Timer(lambda: eval_once(method))
    print(t.timeit(number=cnt) / cnt)
