#include "handle.hpp"
#include "config.hpp"

int mtk::oztcecgemm::create(
		mtk::oztcecgemm::handle_t *h
		) {
	auto handle = (*h = new mtk::oztcecgemm::handle);
	// Initialize cuBLAS handler
	CUTF_CHECK_ERROR(cublasCreate(&(handle->cublas_handle)));

	// Initialize SHGEMM handler
	mtk::shgemm::create(handle->shgemm_handle);

	return 0;
}

int mtk::oztcecgemm::destroy(
		mtk::oztcecgemm::handle_t handle
		) {
	// Destroy cuBLAS handler
	CUTF_CHECK_ERROR(cublasDestroy(handle->cublas_handle));

	// Destroy SHGEMM handler
	mtk::shgemm::destroy(handle->shgemm_handle);

	delete handle;

	return 0;
}

void mtk::oztcecgemm::set_cuda_stream(
		mtk::oztcecgemm::handle_t handle,
		cudaStream_t cuda_stream
		) {
	// Set cuda stream to cuBLAS handler
	CUTF_CHECK_ERROR(cublasSetStream(handle->cublas_handle, cuda_stream));

	// Set cuda stream to SHGEMM handler
	mtk::shgemm::set_cuda_stream(handle->shgemm_handle, cuda_stream);

	// Set oztcecgemm handler
	handle->cuda_stream = cuda_stream;
}

// working memory size calculation
namespace {
std::size_t calculate_working_memory_size(
		const std::size_t m,
		const std::size_t n,
		const mtk::oztcecgemm::compute_mode_t compute_mode,
		const mtk::oztcecgemm::detail::matrix_t matrix
		) {
	const auto split_config = mtk::oztcecgemm::detail::get_split_config(compute_mode);

	decltype(split_config.matrix_a_split_types) split_data_types;
	if (matrix == mtk::oztcecgemm::detail::matrix_A) {
		split_data_types = split_config.matrix_a_split_types;
	} else {
		split_data_types = split_config.matrix_b_split_types;
	}

	std::size_t sum_data_type_size = 0;
	for (const auto d : split_data_types) {
		sum_data_type_size += mtk::oztcecgemm::detail::get_data_size_in_byte(d);
	}

	return sum_data_type_size * m * n;
}
} // unnamed namespace

void mtk::oztcecgemm::reallocate_working_memory(
		mtk::oztcecgemm::handle_t handle,
		const std::vector<std::tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>> gemm_list
		) {
	std::size_t max_working_memory_size = 0;

	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		const auto mode = std::get<3>(gemm);

		const auto working_memory_A = calculate_working_memory_size(m, k, mode, detail::matrix_A);
		const auto working_memory_B = calculate_working_memory_size(k, n, mode, detail::matrix_B);

		max_working_memory_size = std::max(
				max_working_memory_size,
				working_memory_A + working_memory_B
				);
	}

	if (max_working_memory_size > handle->current_working_memory_size) {
		handle->current_working_memory_size = max_working_memory_size;
		cudaFree(handle->working_memory_ptr);

		// Realloc
		cudaMalloc(&(handle->working_memory_ptr), handle->current_working_memory_size);
	}
}

std::string mtk::oztcecgemm::get_compute_mode_name_str(
		const mtk::oztcecgemm::compute_mode_t mode
		) {
	switch (mode) {
	case mtk::oztcecgemm::fp32_split_3:
		return "fp32_split_3";
	default:
		break;
	}
	return "Unknown";
}
