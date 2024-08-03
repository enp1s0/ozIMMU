cmake_minimum_required(VERSION 3.18)
project(ozimmu LANGUAGES CXX CUDA)

set(BUILD_TEST OFF)

find_package(CUDAToolkit 12.1 REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
  set(CMAKE_${lang}_STANDARD          17)
  set(CMAKE_${lang}_STANDARD_REQUIRED ON)
  set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 90 89 80 86 75)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/ozimmu/*.hpp")

SET(lib_name ozimmu)
add_library(${lib_name} SHARED
  ${SRCDIR}/handle.cu
  ${SRCDIR}/config.cu
  ${SRCDIR}/split.cu
  ${SRCDIR}/gemm.cu
  ${SRCDIR}/cublas.cu
  ${SRCDIR}/culip.cu
  ${HEADERS}
)

target_include_directories(${lib_name} PUBLIC ${INCDIR})
target_include_directories(${lib_name} PUBLIC ${SRCDIR}/cutf/include)

target_link_libraries(${lib_name} PRIVATE
  cuda
)

set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER ${INCDIR}/ozimmu/${library}.hpp)
target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

##########################################################################
# Installing
##########################################################################
install(TARGETS ozimmu
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/ozimmu
)

##########################################################################
# Tests
##########################################################################
if (${BUILD_TEST})
  # Directory
  set(TESTSRCDIR test)

  # Check commit
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${TESTSRCDIR}/matfile
    OUTPUT_VARIABLE CURRENT_MATFILE_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(REQUIRED_MATFILE_COMMIT "21b58dd59d3df351f69bc2e7518a81da2664790c")
  execute_process(
    COMMAND git merge-base --is-ancestor ${CURRENT_MATFILE_COMMIT} ${REQUIRED_MATFILE_COMMIT}
    WORKING_DIRECTORY ${TESTSRCDIR}/matfile
    RESULT_VARIABLE IS_OLD_COMMIT
  )
  if(IS_OLD_COMMIT EQUAL 0)
    message(FATAL_ERROR "The matfile submodule is too old. Please run `git submodule update` to update the submodule")
  endif()

  add_subdirectory(test/mateval)
  add_subdirectory(test/gpu_monitor)

  foreach(test main)
    set(test_out ${test}.test)
    add_executable(${test_out} ${TESTSRCDIR}/${test}_test.cu ${HEADERS})
    target_include_directories(${test_out} PRIVATE ${INCDIR})
    target_include_directories(${test_out} PRIVATE ${TESTSRCDIR}/mateval/include)
    target_include_directories(${test_out} PRIVATE ${TESTSRCDIR}/matfile/include)
    target_link_libraries(${test_out} PRIVATE ozimmu)
    target_link_libraries(${test_out} PRIVATE
      CUDA::cublas
      CUDA::curand
      mateval_cuda
      gpu_monitor
    )
  endforeach()
endif()
