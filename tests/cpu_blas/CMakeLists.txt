if(MATHPRIM_ENABLE_BLAS)
    add_executable(cpu_blas main.cpp)
    target_link_libraries(cpu_blas PRIVATE mathprim  GTest::gtest_main)
endif()