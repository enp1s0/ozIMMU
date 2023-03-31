#include <cutf/device.hpp>
#include "handle.hpp"
#include "config.hpp"
#include "utils.hpp"

int mtk::ozimma::create(
		mtk::ozimma::handle_t *h,
		mtk::ozimma::malloc_mode_t mm
		) {
	ozIMMA_log("Initializing ozIMMA handle");
	auto handle = (*h = new mtk::ozimma::handle);
	// Initialize cuBLAS handler
	CUTF_CHECK_ERROR(cublasCreate_org(&(handle->cublas_handle)));

	handle->current_working_memory_size = 0;
	handle->working_memory_ptr = nullptr;
	handle->malloc_mode = mm;

	// Disable profiling by default
	mtk::ozimma::disable_profiling(*h);

	CUTF_CHECK_ERROR(cudaMalloc(&(handle->d_mantissa_loss_counter_ptr), sizeof(unsigned long long int) * handle->mantissa_loss_counter_length));

	return 0;
}

int mtk::ozimma::destroy(
		mtk::ozimma::handle_t handle
		) {
	if (handle) {
		ozIMMA_log("Destroying ozIMMA handle");
		// Destroy cuBLAS handler
		CUTF_CHECK_ERROR(cublasDestroy_org(handle->cublas_handle));

		CUTF_CHECK_ERROR(cudaFree(handle->working_memory_ptr));
		handle->working_memory_ptr = nullptr;

		CUTF_CHECK_ERROR(cudaFree(handle->d_mantissa_loss_counter_ptr));
		handle->d_mantissa_loss_counter_ptr = nullptr;

		delete handle;
		handle = nullptr;
	}

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

std::size_t mtk::ozimma::reallocate_working_memory(
		mtk::ozimma::handle_t handle,
		const mtk::ozimma::gemm_list_t gemm_list
		) {
	std::size_t max_working_memory_size = 0;

	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		const auto element_kind = std::get<3>(gemm);
		const auto mode = std::get<4>(gemm);

		const auto working_memory_A = mtk::ozimma::detail::calculate_working_memory_size(m, k, mode, detail::matrix_A, element_kind);
		const auto working_memory_B = mtk::ozimma::detail::calculate_working_memory_size(k, n, mode, detail::matrix_B, element_kind);
		const auto working_memory_C_fp32 = m * n * mtk::ozimma::get_data_size_in_byte(fp32);
		const auto working_memory_C_fp64 = m * n * mtk::ozimma::get_data_size_in_byte(fp64) * (element_kind == mtk::ozimma::real ? 1 : 2);
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
			etc = (m + n) * mtk::ozimma::get_data_size_in_byte(fp64) * (element_kind == mtk::ozimma::real ? 1 : 2);
		}

		max_working_memory_size = std::max(
				max_working_memory_size,
				working_memory_A + working_memory_B + working_memory_C_fp32 + working_memory_C_fp64 + etc
				);
	}

	if (max_working_memory_size > handle->current_working_memory_size) {
		handle->current_working_memory_size = max_working_memory_size;

		ozIMMA_log("Reallocated moery : " + std::to_string(max_working_memory_size) + " B");

		if (handle->working_memory_ptr != nullptr) {
			if (handle->malloc_mode == mtk::ozimma::malloc_sync) {
				CUTF_CHECK_ERROR(cudaFree(handle->working_memory_ptr));
			} else {
				CUTF_CHECK_ERROR(cudaFreeAsync(handle->working_memory_ptr, handle->cuda_stream));
			}
		}

		// Realloc
		if (handle->malloc_mode == mtk::ozimma::malloc_sync) {
			CUTF_CHECK_ERROR(cudaMalloc(&(handle->working_memory_ptr), handle->current_working_memory_size));
		} else {
			CUTF_CHECK_ERROR(cudaMallocAsync(&(handle->working_memory_ptr), handle->current_working_memory_size, handle->cuda_stream));
		}

		return max_working_memory_size;
	}
	return 0;
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
	case mtk::ozimma::fp64_int8_auto:
		return "fp64_int8_auto";
	default:
		break;
	}
	OZIMMA_NOT_IMPLEMENTED;
	return "";
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
	case mtk::ozimma::fp64_int8_auto:
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

void mtk::ozimma::print_profiler_result(mtk::ozimma::handle_t handle, const std::string tag, const bool csv) {
	if (!csv) {
		handle->profiler.print_result(tag);
	} else {
		handle->profiler.print_result_csv(tag);
	}
}

void mtk::ozimma::clear_profiler_result(mtk::ozimma::handle_t handle) {
	handle->profiler.clear();
}
