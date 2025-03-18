find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

file(GLOB_RECURSE PYMP_SRC "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

if(MATHPRIM_ENABLE_CUDA)
  file(GLOB_RECURSE PYMP_CU "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
  list(APPEND PYMP_SRC ${PYMP_CU})

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  enable_language(CUDA)
endif()

nanobind_add_module(libpymp ${PYMP_SRC})

target_link_libraries(libpymp PRIVATE mathprim)

# Install the module
install(TARGETS libpymp LIBRARY DESTINATION .)