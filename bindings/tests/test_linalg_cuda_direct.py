import numpy as np
from scipy.sparse import csr_matrix
from pymp.linalg import (
    cg_cuda_csr_direct,
    pcg_cuda_csr_direct_diagonal,
    pcg_cuda_csr_direct_ic,
    pcg_cuda_csr_direct_ainv,
    pcg_cuda_csr_direct_with_ext_spai,
    grid_laplacian_nd_dbc,
    ainv,
)
import torch
from timeit import Timer

all_methods = [
    cg_cuda_csr_direct,
    pcg_cuda_csr_direct_diagonal,
    pcg_cuda_csr_direct_ic,
    pcg_cuda_csr_direct_ainv,
]
dtype = torch.float32
# 3 1 0
# 1 3 1
# 0 1 3
# Store in CSR
outer_ptrs = torch.tensor([0, 2, 5, 7], dtype=torch.int32, device=torch.device('cuda'))
inner_ptrs = torch.tensor([0, 1, 0, 1, 2, 1, 2], dtype=torch.int32, device=torch.device('cuda'))
values = torch.tensor([3, 1, 1, 3, 1, 1, 3], dtype=dtype, device=torch.device('cuda'))
b = torch.tensor([1, 2, 3], dtype=dtype, device=torch.device('cuda'))

for m in all_methods:
    x = torch.zeros(3, dtype=dtype, device=torch.device('cuda'))
    print(m.__name__, m(
      outer_ptrs=outer_ptrs,
      inner_indices=inner_ptrs,
      values=values,
      rows=3, cols=3,
      b=b,
      x=x,
      max_iter=100,
      verbose=1
    ))
#### pcg_cuda_csr_direct_with_ext_spai
mat = csr_matrix(([3, 1, 1, 3, 1, 1, 3], ([0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2])), shape=(3, 3), dtype=np.float32)
ai = ainv(mat)
print(ai)
ai_rowptrs = torch.tensor(ai.indptr, dtype=torch.int32, device=torch.device('cuda'))
ai_colinds = torch.tensor(ai.indices, dtype=torch.int32, device=torch.device('cuda'))
ai_values = torch.tensor(ai.data, dtype=dtype, device=torch.device('cuda'))
# ai_torch = torch.sparse_csr_tensor(
#     torch.tensor(ai.data, dtype=dtype, device=torch.device("cuda")),
#     torch.tensor(ai.indices, dtype=torch.int32, device=torch.device("cuda")),
#     torch.tensor(ai.indptr, dtype=torch.int32, device=torch.device("cuda")),
#     mat.shape,
# )
x = torch.zeros(3, dtype=dtype, device=torch.device('cuda'))
b = torch.tensor([1, 2, 3], dtype=dtype, device=torch.device('cuda'))
pcg_cuda_csr_direct_with_ext_spai(
    outer_ptrs=outer_ptrs,
    inner_indices=inner_ptrs,
    values=values,
    rows=3, cols=3,
    ainv_outer_ptrs=ai_rowptrs,
    ainv_inner_indices=ai_colinds,
    ainv_values=ai_values, epsilon=0,
    b=b,
    x=x,
    max_iter=100,
    verbose=1
)

N = 1024
laplacian = grid_laplacian_nd_dbc([N, N])
outer = torch.tensor(laplacian.indptr, dtype=torch.int32, device=torch.device('cuda'))
inner = torch.tensor(laplacian.indices, dtype=torch.int32, device=torch.device('cuda'))
values = torch.tensor(laplacian.data, dtype=dtype, device=torch.device('cuda'))
b = torch.tensor(laplacian @ np.ones(N * N, dtype=np.float32), dtype=dtype, device=torch.device('cuda'))

def eval_once(m):
    x = torch.zeros_like(b)
    name = m.__name__
    print(name, m(
        outer_ptrs=outer,
        inner_indices=inner,
        values=values,
        rows=N * N, cols=N * N,
        b=b,
        x=x,
        rtol=1e-6,
        max_iter=N * N,
    ))

cnt = 5
for m in all_methods:
    t = Timer(lambda: eval_once(m))
    te = t.timeit(number=cnt)
    print(f"{m.__name__} took {te / cnt:.3f} seconds on average")
