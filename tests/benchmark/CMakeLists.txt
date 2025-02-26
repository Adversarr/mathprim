macro(mathprim_add_benchmark name)
  add_executable(BM_${name} ${name}.cpp)
  target_link_libraries(BM_${name} PRIVATE benchmark::benchmark mathprim)
  target_compile_options(BM_${name} PRIVATE -march=native)
endmacro(mathprim_add_benchmark)

find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)

mathprim_add_benchmark(map)
mathprim_add_benchmark(blas)
mathprim_add_benchmark(sparse_csr)
mathprim_add_benchmark(par)
mathprim_add_benchmark(csr_cg)

# ########## CUDA benchmarks ###########
if(MATHPRIM_ENABLE_CUDA)
  macro(mathprim_add_benchmark name)
    add_executable(BM_cu_${name} ${name}.cu)
    target_link_libraries(BM_cu_${name} PRIVATE benchmark::benchmark mathprim)
    target_compile_options(BM_cu_${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
  endmacro()

  mathprim_add_benchmark(blas)
  mathprim_add_benchmark(par)
  mathprim_add_benchmark(csr_cg)
  mathprim_add_benchmark(csr_fp)
endif()
