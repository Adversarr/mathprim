mathprim_add_benchmark(map)
mathprim_add_benchmark(blas)
mathprim_add_benchmark(sparse_csr)
mathprim_add_benchmark(par)
mathprim_add_benchmark(csr_cg)

# ########## CUDA benchmarks ###########
if(MATHPRIM_ENABLE_CUDA)
  mathprim_add_benchmark_cu(blas)
  mathprim_add_benchmark_cu(par)
  mathprim_add_benchmark_cu(csr_cg)
  mathprim_add_benchmark_cu(sparse_csr)
  mathprim_add_benchmark_cu(csr_fp)
endif()
