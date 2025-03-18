from pymp.linalg.cholmod import chol, is_cholmod_available
from pymp.linalg import ldlt, llt
import numpy as np
from scipy.sparse import csr_matrix
from timeit import Timer
x = np.array([1,2,3], dtype=np.float32)
b = np.array([3,2,1], dtype=np.float32)
A = csr_matrix(
  [[3, 1, 0],
   [1, 3, 1],
   [0, 1, 3]], dtype=np.float32)

meth_list =  [ldlt, llt]
if is_cholmod_available():
    meth_list.append(chol)

for method in meth_list:
    algorithm = method(A)
    print(algorithm)
    algorithm.solve(b, x)
    print(A @ x - b)

x = x.astype(np.float64)
b = b.astype(np.float64)
A = A.astype(np.float64)

for method in meth_list:
    algorithm = method(A)
    print(algorithm)
    algorithm.solve(b, x)
    print(A @ x - b)