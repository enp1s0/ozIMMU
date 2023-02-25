#pragma once
#include <oztcecgemm/oztcecgemm.hpp>
#include <cutf/cublas.hpp>
#include <cutf/debug/time_breakdown.hpp>

struct mtk::oztcecgemm::handle {
	// handlers
	cublasHandle_t cublas_handle;
	cudaStream_t cuda_stream;

	// working memory
	void* working_memory_ptr;
	std::size_t current_working_memory_size;

	// profiling
	cutf::debug::time_breakdown::profiler profiler;
};
