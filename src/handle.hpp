#pragma once
#include <oztcecgemm/oztcecgemm.hpp>
#include <shgemm/shgemm.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include <cutf/cublas.hpp>
#include <cutf/debug/time_breakdown.hpp>

struct mtk::oztcecgemm::handle {
	// handlers
	cublasHandle_t cublas_handle;
	mtk::shgemm::shgemmHandle_t shgemm_handle;
	cudaStream_t cuda_stream;
	cumpsgemm::handle_t cumpsgemm_handle;

	// working memory
	void* working_memory_ptr;
	std::size_t current_working_memory_size;

	// profiling
	cutf::debug::time_breakdown::profiler profiler;
};
