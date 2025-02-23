find_package(benchmark REQUIRED)
add_executable(nystorm main.cpp)
target_link_libraries(nystorm PRIVATE mathprim benchmark::benchmark)

if(MATHPRIM_ENABLE_CUDA)
  add_executable(cu_nystorm main.cu)
  target_link_libraries(cu_nystorm PRIVATE mathprim benchmark::benchmark)
endif()