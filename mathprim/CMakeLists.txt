add_library(mathprim INTERFACE)
target_include_directories(mathprim INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MATHPRIM_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
  else()
    target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
  endif()
endif()

# Eigen is a must.
find_package(Eigen3 REQUIRED)
message(STATUS "Enabling Eigen")
target_link_libraries(mathprim INTERFACE Eigen3::Eigen)

if(MATHPRIM_ENABLE_BLAS)
  if(MATHPRIM_ENABLE_MKL)
    find_package(MKL CONFIG REQUIRED)
    set(BLA_VENDOR "Intel" CACHE STRING "BLAS vendor")
    message(STATUS "Enabling MKL as BLAS")
    target_link_libraries(mathprim INTERFACE MKL::MKL)
    target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_INTEL_MKL)
    target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_MKL)
  else()
    if(APPLE)
      # Prefer to use Accelerate framework on macOS
      set(BLA_VENDOR "Apple" CACHE STRING "BLAS vendor")
      target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_APPLE)
      target_compile_definitions(mathprim INTERFACE ACCELERATE_NEW_LAPACK)
    else()
      set(BLA_VENDOR "OpenBLAS" CACHE STRING "BLAS vendor")
      target_compile_definitions(mathprim INTERFACE MATHPRIM_BLAS_VENDOR_OPENBLAS)
    endif()

    find_package(BLAS REQUIRED)
    message(STATUS "Enabling BLAS:: (${BLA_VENDOR}) -> ${BLAS_LIBRARIES}")

    target_link_libraries(mathprim INTERFACE ${BLAS_LIBRARIES})
  endif()

  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_BLAS)
else()
  if(MATHPRIM_ENABLE_MKL)
    message(FATAL_ERROR "MKL cannot be enabled without BLAS")
  endif()
endif()

if(MATHPRIM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Enabling CUDA")
  message(STATUS "CUDA tk version: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA tk path: ${CUDAToolkit_ROOT}")

  if(MATHPRIM_CUDA_LINK_DYNAMIC)
    message(STATUS "Linking to dynamic CUDA libraries. Please check if your application support this.")
    target_link_libraries(mathprim INTERFACE CUDA::cudart CUDA::cublas CUDA::cusparse)
  else()
    target_link_libraries(mathprim INTERFACE
      CUDA::cudart_static
      CUDA::cublas_static
      CUDA::cusparse_static)
  endif()
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_CUDA)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  # OpenMP
  if(MATHPRIM_ENABLE_OPENMP)
    if(MSVC)
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler:/openmp>)
    else()
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    endif()
  endif()

  # Warnings:
  if(MATHPRIM_ENABLE_WARNINGS)
    if(MSVC)
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)

    # target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Wcross-execution-space-call>)
    else()
      target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

      # target_compile_options(mathprim INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-W=cross-execution-space-call>)
    endif()
  endif()
endif()

if(MATHPRIM_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  message(STATUS "Enabling OpenMP")

  target_link_libraries(mathprim INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_OPENMP)
endif()

if(MATHPRIM_ENABLE_CHOLMOD)
  find_package(CHOLMOD REQUIRED)
  message(STATUS "Enabling CHOLMOD")

  target_link_libraries(mathprim INTERFACE SuiteSparse::CHOLMOD)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_CHOLMOD)
endif()

if (MATHPRIM_ENABLE_SUPERLU)
  find_package(superlu REQUIRED)
  message(STATUS "Enabling SUPERLU")

  target_link_libraries(mathprim INTERFACE superlu::superlu)
  target_compile_definitions(mathprim INTERFACE MATHPRIM_ENABLE_SUPERLU)
endif()

if(MATHPRIM_INSTALL)
  include(GNUInstallDirs)
  install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh")

  install(TARGETS mathprim
    EXPORT mathprimTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(EXPORT mathprimTargets
    FILE mathprimTargets.cmake
    NAMESPACE mathprim::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mathprim)

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/../cmake/mathprimConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mathprim
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfigVersion.cmake
    VERSION 0.0.1
    COMPATIBILITY AnyNewerVersion
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mathprimConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mathprim
  )
endif()
