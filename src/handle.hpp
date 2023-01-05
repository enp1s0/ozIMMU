#pragma once
#include <oztcecgemm/oztcecgemm.hpp>
#include <shgemm/shgemm.hpp>
#include <cutf/cublas.hpp>

struct mtk::oztcecgemm::handle_t {
	// handlers
	cublasHandle_t cublas_handle;
	mtk::shgemm::shgemmHandle_t shgemm_handle;
	cudaStream_t cuda_stream;

	// working memory
	void* working_memory_ptr;
	std::size_t current_working_memory_size;
};
