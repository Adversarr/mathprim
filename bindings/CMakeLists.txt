find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE PYMP_SRC "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

if(MATHPRIM_ENABLE_CUDA)
  file(GLOB_RECURSE PYMP_CU "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
  list(APPEND PYMP_SRC ${PYMP_CU})

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  enable_language(CUDA)
endif()

set(PYMATHPRIM_CXX_FLAGS "")

# disable warning: ISO C++ forbids zero-size array
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND PYMATHPRIM_CXX_FLAGS "-Wno-zero-length-array")
endif()

nanobind_add_module(libpymathprim ${PYMP_SRC})

target_link_libraries(libpymathprim PRIVATE mathprim)
target_compile_options(libpymathprim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${PYMATHPRIM_CXX_FLAGS}>)
# for cuda
foreach(flags IN LISTS PYMATHPRIM_CXX_FLAGS)
  target_compile_options(libpymathprim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flags}>)
endforeach()


# Install the module
install(TARGETS libpymathprim LIBRARY DESTINATION .)