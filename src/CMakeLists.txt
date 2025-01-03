add_library(mathprim INTERFACE)
target_include_directories(mathprim INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

if(MATHPRIM_ENABLE_WARNINGS)
  target_compile_options(mathprim INTERFACE -Wall)
endif()
