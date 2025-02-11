add_executable(sparse_csr main.cpp)
target_link_libraries(sparse_csr PRIVATE mathprim GTest::GTest GTest::Main)
add_test(NAME sparse_csr COMMAND sparse_csr)