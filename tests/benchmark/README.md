# Benchmarkings

## Solving Sparse System

Problem setup: N = 512x512, 2D laplacian with DBC.

We compare mathprim to SCIPY and CUPY, See implementation code [here](cu_csr.py). 

```
scipy::factorize: 0.799168 sec
scipy::solve: 0.035988 sec
cupy::factorize: 1.032221 sec
cupy::solve: 0.880213 sec
scipy::cg: 0.943054 sec
cupy::cg: 0.233852 sec
scipy::cg iter= 1104
cupy::cg iter= 1044
Done.
```

mathprim implementation: run `BM_csr_cg` and `BM_cu_csr_cg`.

```
--- BM_csr_cg
work<blas::cpu_blas<float>>/512              310 ms
work<blas::cpu_eigen<float>>/512             302 ms 

--- BM_cu_csr_cg
work_cuda/512                127 ms
work_cuda_no_prec/512        123 ms
work_cuda_ilu0/512          1012 ms
work_cuda_ic/512            1008 ms
work_cuda_ai/512             104 ms

--- ... enlarge to 1024x1024 problem
work_cuda/1024                853 ms
work_cuda_no_prec/1024        794 ms
work_cuda_ic/1024            5944 ms
work_cuda_ai/1024             669 ms
```

## SpMV

On the same problem, and matrix. Torch's default sparse csr [implementation](spmv.py).

```
### 512
spmv takes 0.119803 ms.
spmm takes 0.835651 ms.
### 1024
spmv takes 0.362839 ms.
spmm takes 3.750413 ms.
```

mathprim's:
```
work_cuda/512            0.070 ms        0.070 ms
work_cuda/1024           0.233 ms        0.233 ms
work_cuda_spmm/512       0.512 ms        0.510 ms
work_cuda_spmm/1024       2.02 ms         2.01 ms
```