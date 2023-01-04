cmake_minimum_required(VERSION 3.18)
project(oztcecgemm LANGUAGES CXX CUDA)

set(BUILD_TEST OFF)

find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()

# CUDA (1/2)
SET(CMAKE_CUDA_ARCHITECTURES 80 86)

# Directories
set(INCDIR include)
set(SRCDIR src)

file(GLOB HEADERS "${INCDIR}/oztcecgemm/*.hpp")

SET(lib_name oztcecgemm)
add_library(${lib_name} STATIC
	${SRCDIR}/handle.cu
	${SRCDIR}/config.cu
	${SRCDIR}/split.cu
	${SRCDIR}/shgemm/src/shgemm.cu
	${HEADERS}
	)

target_include_directories(${lib_name} PUBLIC ${INCDIR})
target_include_directories(${lib_name} PUBLIC ${SRCDIR}/cutf/include)
target_include_directories(${lib_name} PUBLIC ${SRCDIR}/shgemm/include)
target_include_directories(${lib_name} PUBLIC ${SRCDIR}/shgemm/src/wmma_extension/include)

target_link_libraries(${lib_name} PRIVATE
	cuda
	)

set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER ${INCDIR}/oztcecgemm/${library}.hpp)
target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

##########################################################################
# Installing
##########################################################################
install(TARGETS oztcecgemm
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/oztcecgemm
	)

##########################################################################
# Tests
##########################################################################
if (${BUILD_TEST})
	# Directory
	set(TESTSRCDIR test)

	foreach(test shgemm shgemm_k_slicing exponent power_consumption)
		set(test_out ${test}.test)
		add_executable(${test_out} ${TESTSRCDIR}/${test}_test.cu ${HEADERS})
		target_include_directories(${test_out} PRIVATE ${INCDIR})
		target_include_directories(${test_out} PRIVATE ${TESTSRCDIR}/mateval/include ${TESTSRCDIR}/gpu_monitor/include)
		target_link_directories(${test_out} PRIVATE ${TESTSRCDIR}/mateval/build ${TESTSRCDIR}/gpu_monitor/build)
		target_link_libraries(${test_out} PRIVATE mateval_cuda)
		target_link_libraries(${test_out} PRIVATE gpu_monitor)
		target_link_libraries(${test_out} PRIVATE shgemm)
		target_link_libraries(${test_out} PRIVATE
			CUDA::cublas
			CUDA::curand
			nvidia-ml
			)
	endforeach()
endif()
