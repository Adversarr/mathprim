find_package(BLAS)
add_executable(cpu_blas main.cpp)
target_link_libraries(cpu_blas PRIVATE mathprim)

if(BLAS_FOUND)
    message(STATUS "BLAS found")
    target_link_libraries(cpu_blas PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(cpu_blas PUBLIC "MATHPRIM_BLAS_FOUND=1")
endif()