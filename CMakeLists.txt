cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(
    FATAL_ERROR
    "In-source builds not allowed. Please make a new directory (called
      a build directory) and run CMake from there. You may need to
      remove CMakeCache.txt. "
  )
endif()

# master project detection with cmake variable
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MATHPRIM_MASTER_PROJECT ON)
else()
  set(MATHPRIM_MASTER_PROJECT OFF)
endif()

project(mathprim)

# Test if cuda is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  set(MATHPRIM_ENABLE_CUDA_TEST ${MATHPRIM_MASTER_PROJECT})
else()
  set(MATHPRIM_ENABLE_CUDA_TEST OFF)
endif()

option(MATHPRIM_ENABLE_WARNINGS "Enable compiler warnings" ${MATHPRIM_MASTER_PROJECT})
option(MATHPRIM_ENABLE_EIGEN "Enable Eigen" OFF)
option(MATHPRIM_ENABLE_BLAS "Enable BLAS" OFF)
option(MATHPRIM_ENABLE_OPENMP "Enable OpenMP" ON)
option(MATHPRIM_ENABLE_CUDA "Enable CUDA" ${MATHPRIM_ENABLE_CUDA_TEST})
option(MATHPRIM_BUILD_TESTS "Build tests" ${MATHPRIM_MASTER_PROJECT})
option(MATHPRIM_BUILD_TESTS_CUDA "Build cuda tests" ${MATHPRIM_ENABLE_CUDA_TEST})

if(MATHPRIM_BUILD_TESTS_CUDA)
  if (NOT MATHPRIM_ENABLE_CUDA)
    message(FATAL_ERROR "CUDA tests require CUDA to be enabled")
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES 86)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  enable_language(CUDA)
endif()

add_subdirectory(src)

if(MATHPRIM_BUILD_TESTS)
  enable_testing()
  message(STATUS "Building tests")
  add_subdirectory(tests)
endif()
