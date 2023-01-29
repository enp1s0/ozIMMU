#include "handle.hpp"
#include "config.hpp"
#include "utils.hpp"

int mtk::oztcecgemm::create(
		mtk::oztcecgemm::handle_t *h
		) {
	auto handle = (*h = new mtk::oztcecgemm::handle);
	// Initialize cuBLAS handler
	CUTF_CHECK_ERROR(cublasCreate(&(handle->cublas_handle)));

	// Initialize SHGEMM handler
	mtk::shgemm::create(handle->shgemm_handle);

	// Initialize cuMpSGEMM handler
	cumpsgemm::create(handle->cumpsgemm_handle);

	// Disable profiling by default
	mtk::oztcecgemm::disable_profiling(*h);

	return 0;
}

int mtk::oztcecgemm::destroy(
		mtk::oztcecgemm::handle_t handle
		) {
	// Destroy cuBLAS handler
	CUTF_CHECK_ERROR(cublasDestroy(handle->cublas_handle));

	// Destroy SHGEMM handler
	mtk::shgemm::destroy(handle->shgemm_handle);

	// Destroy cuMpSGEMM handler
	cumpsgemm::destroy(handle->cumpsgemm_handle);

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

	// Set cuda stream to cuMpSGEMM handler
	cumpsgemm::set_stream(handle->cumpsgemm_handle, cuda_stream);

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

	decltype(split_config.matrix_A_split_types) split_data_types;
	if (matrix == mtk::oztcecgemm::detail::matrix_A) {
		split_data_types = split_config.matrix_A_split_types;
	} else {
		split_data_types = split_config.matrix_B_split_types;
	}

	std::size_t sum_data_type_size = 0;
	for (const auto d : split_data_types) {
		sum_data_type_size += mtk::oztcecgemm::get_data_size_in_byte(d);
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
		const auto working_memory_C_fp32 = m * n * mtk::oztcecgemm::get_data_size_in_byte(fp32);
		const auto working_memory_C_fp64 = m * n * mtk::oztcecgemm::get_data_size_in_byte(fp64);
		std::size_t etc = 0;
		if (
				mode == mtk::oztcecgemm::fp64_int8_6 ||
				mode == mtk::oztcecgemm::fp64_int8_7 ||
				mode == mtk::oztcecgemm::fp64_int8_8 ||
				mode == mtk::oztcecgemm::fp64_int8_9
			 ) {
			etc = (m + n) * mtk::oztcecgemm::get_data_size_in_byte(fp64);
		}

		max_working_memory_size = std::max(
				max_working_memory_size,
				working_memory_A + working_memory_B + working_memory_C_fp32 + working_memory_C_fp64 + etc
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
	case mtk::oztcecgemm::sgemm:
		return "sgemm";
	case mtk::oztcecgemm::fp64_int8_6:
		return "fp64_int8_6";
	case mtk::oztcecgemm::fp64_int8_7:
		return "fp64_int8_7";
	case mtk::oztcecgemm::fp64_int8_8:
		return "fp64_int8_8";
	case mtk::oztcecgemm::fp64_int8_9:
		return "fp64_int8_9";
	default:
		break;
	}
	return "Unknown";
}

mtk::oztcecgemm::data_t mtk::oztcecgemm::get_output_type(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::oztcecgemm::sgemm:
		return mtk::oztcecgemm::fp32;

	case mtk::oztcecgemm::fp32_split_3:
	case mtk::oztcecgemm::fp64_int8_6:
	case mtk::oztcecgemm::fp64_int8_7:
	case mtk::oztcecgemm::fp64_int8_8:
	case mtk::oztcecgemm::fp64_int8_9:
		return mtk::oztcecgemm::fp64;

	default:
		break;
	}
	OZTCECGEM_NOT_IMPLEMENTED;
	return mtk::oztcecgemm::original;
}

std::size_t mtk::oztcecgemm::get_data_size_in_byte(
		const mtk::oztcecgemm::data_t d
		) {
	switch (d) {
	case mtk::oztcecgemm::fp64:
		return 8;
	case mtk::oztcecgemm::fp32:
		return 4;
	case mtk::oztcecgemm::fp16:
		return 2;
	case mtk::oztcecgemm::original:
		return 0;
	case mtk::oztcecgemm::int8:
		return 1;
	default:
		break;
	}
	return 0;
}

void mtk::oztcecgemm::enable_profiling(mtk::oztcecgemm::handle_t handle) {
	handle->profiler.enable_measurement();
}

void mtk::oztcecgemm::disable_profiling(mtk::oztcecgemm::handle_t handle) {
	handle->profiler.disable_measurement();
}

void mtk::oztcecgemm::print_profiler_result(mtk::oztcecgemm::handle_t handle, const bool csv) {
	if (!csv) {
		handle->profiler.print_result();
	} else {
		handle->profiler.print_result_csv();
	}
}

void mtk::oztcecgemm::clear_profiler_result(mtk::oztcecgemm::handle_t handle) {
	handle->profiler.clear();
}
