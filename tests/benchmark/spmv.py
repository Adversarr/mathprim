from argparse import ArgumentParser
from scipy.sparse import csr_matrix
import torch
import numpy as np
from timeit import Timer
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
    return torch.sparse_coo_tensor(torch.tensor([row, col], dtype=torch.int32), torch.tensor(data, dtype=torch.float32), (n * n, n * n)).to_sparse_csr()

parser = ArgumentParser()
parser.add_argument('--n', type=int, default=128)
parser.add_argument('--cnt', type=int, default=1)

args = parser.parse_args()
n = args.n
cnt = args.cnt
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

matrix = propose_laplacian2d(n)
print("Matrix Preparation Done.")

def spmv(mat, vec):
    y = mat @ vec
    # sync threads
    torch.cuda.synchronize()


def work(name, fn, *args):
    t = Timer(lambda: fn(*args))
    avg_time = t.timeit(number=cnt) / cnt * 1000
    print(f"{name} takes {avg_time:.6f} ms.")

for i in range(10):
    spmv(matrix, torch.randn(n * n))
    spmv(matrix, torch.randn(n * n, 32))

work("spmv", spmv, matrix, torch.randn(n * n))
work("spmm", spmv, matrix, torch.randn(n * n, 32))