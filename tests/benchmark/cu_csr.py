import cupy as cp
import numpy as np
from argparse import ArgumentParser
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, cg
from cupy.sparse import linalg as cupy_linalg
from timeit import Timer
# cholmod
from sksparse.cholmod import cholesky

def propose_laplacian2d(n):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()
    row = []
    col = []
    data = []
    for i in range(n):
        for j in range(n):
            row.append(i * n + j)
            col.append(i * n + j)
            data.append(4)
            if i > 0:
                row.append(i * n + j)
                col.append((i - 1) * n + j)
                data.append(-1)
            if i < n - 1:
                row.append(i * n + j)
                col.append((i + 1) * n + j)
                data.append(-1)
            if j > 0:
                row.append(i * n + j)
                col.append(i * n + j - 1)
                data.append(-1)
            if j < n - 1:
                row.append(i * n + j)
                col.append(i * n + j + 1)
                data.append(-1)
    return csr_matrix((data, (row, col)), dtype=np.float32, shape=(n * n, n * n))

parser = ArgumentParser()
parser.add_argument('--n', type=int, default=128)
parser.add_argument('--cnt', type=int, default=1)

args = parser.parse_args()
n = args.n
cnt = args.cnt
tol = 1e-6
matrix = propose_laplacian2d(n)
print("Matrix Preparation Done.")
cu_matrix = cp.sparse.csr_matrix(matrix)
cu_lu = cupy_linalg.splu(cu_matrix)
lu = splu(matrix)
print("Factorization Done.")

def do_scipy_cholesky_factorize(mat):
    lu = splu(mat)
def do_scipy_cholesky_solve(lu):
    lu.solve(np.ones(n * n, dtype=np.float32))
def do_cupy_cholesky_factorize(cumat):
    lu = cupy_linalg.splu(cumat)
def do_cupy_cholesky_solve(cu_lu):
    cu_lu.solve(cp.ones(n * n))

def do_scipy_cg(mat):
    count_iter = 0
    def callback(xk):
        nonlocal count_iter
        count_iter += 1
    cg(mat, np.ones(n * n, dtype=np.float32), rtol=tol, callback=callback)
    return count_iter
def do_cupy_cg(cumat):
    count_iter = 0
    def callback(xk):
        nonlocal count_iter
        count_iter += 1
    res, iter = cupy_linalg.cg(cumat, cp.ones(n * n), tol=tol, callback=callback)
    return count_iter

def do_test(name, fn, *args):
    t = Timer(lambda: fn(*args))
    average_cost = t.timeit(number=cnt) / cnt
    print(f'{name}: {average_cost:.6f} sec')

do_test('scipy::factorize', do_scipy_cholesky_factorize, matrix)
do_test('scipy::solve', do_scipy_cholesky_solve, lu)
do_test('cupy::factorize', do_cupy_cholesky_factorize, cu_matrix)
do_test('cupy::solve', do_cupy_cholesky_solve, cu_lu)
do_test('scipy::cg', do_scipy_cg, matrix)
do_test('cupy::cg', do_cupy_cg, cu_matrix)
print("scipy::cg iter=", do_scipy_cg(matrix))
print("cupy::cg iter=", do_cupy_cg(cu_matrix))
print("Done.")