#include <cutf/device.hpp>
#include "handle.hpp"
#include "config.hpp"
#include "utils.hpp"

int mtk::ozimma::create(
		mtk::ozimma::handle_t *h
		) {
	auto handle = (*h = new mtk::ozimma::handle);
	// Initialize cuBLAS handler
	CUTF_CHECK_ERROR(cublasCreate_org(&(handle->cublas_handle)));

	// Disable profiling by default
	mtk::ozimma::disable_profiling(*h);

	return 0;
}

int mtk::ozimma::destroy(
		mtk::ozimma::handle_t handle
		) {
	// Destroy cuBLAS handler
	CUTF_CHECK_ERROR(cublasDestroy_org(handle->cublas_handle));

	delete handle;

	return 0;
}

void mtk::ozimma::set_cuda_stream(
		mtk::ozimma::handle_t handle,
		cudaStream_t cuda_stream
		) {
	// Set cuda stream to cuBLAS handler
	CUTF_CHECK_ERROR(cublasSetStream(handle->cublas_handle, cuda_stream));

	// Set ozimma handler
	handle->cuda_stream = cuda_stream;
}

// working memory size calculation
namespace {
std::size_t calculate_working_memory_size(
		const std::size_t m,
		const std::size_t n,
		const mtk::ozimma::compute_mode_t compute_mode,
		const mtk::ozimma::detail::matrix_t matrix
		) {
	const auto split_config = mtk::ozimma::detail::get_split_config(compute_mode);

	decltype(split_config.matrix_A_split_types) split_data_types;
	if (matrix == mtk::ozimma::detail::matrix_A) {
		split_data_types = split_config.matrix_A_split_types;
	} else {
		split_data_types = split_config.matrix_B_split_types;
	}

	std::size_t sum_data_type_size = 0;
	for (const auto d : split_data_types) {
		sum_data_type_size += mtk::ozimma::get_data_size_in_byte(d);
	}

	return sum_data_type_size * m * n;
}
} // unnamed namespace

void mtk::ozimma::reallocate_working_memory(
		mtk::ozimma::handle_t handle,
		const std::vector<std::tuple<std::size_t, std::size_t, std::size_t, mtk::ozimma::compute_mode_t>> gemm_list
		) {
	std::size_t max_working_memory_size = 0;

	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		const auto mode = std::get<3>(gemm);

		const auto working_memory_A = calculate_working_memory_size(m, k, mode, detail::matrix_A);
		const auto working_memory_B = calculate_working_memory_size(k, n, mode, detail::matrix_B);
		const auto working_memory_C_fp32 = m * n * mtk::ozimma::get_data_size_in_byte(fp32);
		const auto working_memory_C_fp64 = m * n * mtk::ozimma::get_data_size_in_byte(fp64);
		std::size_t etc = 0;
		if (
				mode == mtk::ozimma::fp64_int8_6  ||
				mode == mtk::ozimma::fp64_int8_7  ||
				mode == mtk::ozimma::fp64_int8_8  ||
				mode == mtk::ozimma::fp64_int8_9  ||
				mode == mtk::ozimma::fp64_int8_10 ||
				mode == mtk::ozimma::fp64_int8_11 ||
				mode == mtk::ozimma::fp64_int8_12 ||
				mode == mtk::ozimma::fp64_int8_13
			 ) {
			etc = (m + n) * mtk::ozimma::get_data_size_in_byte(fp64);
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

std::string mtk::ozimma::get_compute_mode_name_str(
		const mtk::ozimma::compute_mode_t mode
		) {
	switch (mode) {
	case mtk::ozimma::sgemm:
		return "sgemm";
	case mtk::ozimma::dgemm:
		return "dgemm";
	case mtk::ozimma::fp64_int8_6:
		return "fp64_int8_6";
	case mtk::ozimma::fp64_int8_7:
		return "fp64_int8_7";
	case mtk::ozimma::fp64_int8_8:
		return "fp64_int8_8";
	case mtk::ozimma::fp64_int8_9:
		return "fp64_int8_9";
	case mtk::ozimma::fp64_int8_10:
		return "fp64_int8_10";
	case mtk::ozimma::fp64_int8_11:
		return "fp64_int8_11";
	case mtk::ozimma::fp64_int8_12:
		return "fp64_int8_12";
	case mtk::ozimma::fp64_int8_13:
		return "fp64_int8_13";
	default:
		break;
	}
	return "Unknown";
}

mtk::ozimma::data_t mtk::ozimma::get_output_type(
		const mtk::ozimma::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::ozimma::sgemm:
		return mtk::ozimma::fp32;

	case mtk::ozimma::fp64_int8_6:
	case mtk::ozimma::fp64_int8_7:
	case mtk::ozimma::fp64_int8_8:
	case mtk::ozimma::fp64_int8_9:
	case mtk::ozimma::fp64_int8_10:
	case mtk::ozimma::fp64_int8_11:
	case mtk::ozimma::fp64_int8_12:
	case mtk::ozimma::fp64_int8_13:
	case mtk::ozimma::dgemm:
		return mtk::ozimma::fp64;

	default:
		break;
	}
	OZIMMA_NOT_IMPLEMENTED;
	return mtk::ozimma::original;
}

std::size_t mtk::ozimma::get_data_size_in_byte(
		const mtk::ozimma::data_t d
		) {
	switch (d) {
	case mtk::ozimma::fp64:
		return 8;
	case mtk::ozimma::fp32:
		return 4;
	case mtk::ozimma::fp16:
		return 2;
	case mtk::ozimma::original:
		return 0;
	case mtk::ozimma::int8:
		return 1;
	default:
		break;
	}
	return 0;
}

void mtk::ozimma::enable_profiling(mtk::ozimma::handle_t handle) {
	handle->profiler.enable_measurement();
}

void mtk::ozimma::disable_profiling(mtk::ozimma::handle_t handle) {
	handle->profiler.disable_measurement();
}

void mtk::ozimma::print_profiler_result(mtk::ozimma::handle_t handle, const bool csv) {
	if (!csv) {
		handle->profiler.print_result();
	} else {
		handle->profiler.print_result_csv();
	}
}

void mtk::ozimma::clear_profiler_result(mtk::ozimma::handle_t handle) {
	handle->profiler.clear();
}
