#include <cutf/cublas.hpp>
#include <shgemm/shgemm.hpp>
#include "config.hpp"
#include "split.hpp"
#include "utils.hpp"
#include "handle.hpp"

namespace {
template <class T>
std::size_t split_core(
		void* const split_ptr,
		const mtk::oztcecgemm::operation_t op,
		const std::size_t m,
		const std::size_t n,
		const T* const src_ptr, const std::size_t ld,
		const std::vector<mtk::oztcecgemm::data_t> data_type_list,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		const T* const two_to_alpha_ptr,
		cudaStream_t cuda_stream
		) {
	const auto num_split = data_type_list.size() - 1;
	std::size_t offset = 0;

	if (num_split <= 1) {
		// Do nothing
	} else if (num_split == 2) {
		mtk::oztcecgemm::split_2(
				reinterpret_cast<std::uint8_t*>(split_ptr), data_type_list[1],
				reinterpret_cast<std::uint8_t*>(split_ptr) + mtk::oztcecgemm::get_data_size_in_byte(data_type_list[1]) * m * n, data_type_list[2],
				m, n,
				src_ptr, mtk::oztcecgemm::detail::get_data_t<T>(), ld,
				op,
				matrix,
				two_to_alpha_ptr,
				cuda_stream
				);
		offset += mtk::oztcecgemm::get_data_size_in_byte(data_type_list[1]) * m * n;
		offset += mtk::oztcecgemm::get_data_size_in_byte(data_type_list[2]) * m * n;
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}

	return offset;
}

template <class T>
void split_AB(
		mtk::oztcecgemm::handle_t handle,
		void* working_memory_ptr,
		const mtk::oztcecgemm::operation_t op_A,
		const mtk::oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	const auto two_to_alpha = mtk::oztcecgemm::get_two_to_alpha<T>(k);

	const auto split_config = mtk::oztcecgemm::detail::get_split_config(compute_mode);

	handle->profiler.start_timer_sync("split_A");
	const auto b_offset = split_core(
			working_memory_ptr,
			op_A,
			m, k,
			a_ptr, lda,
			split_config.matrix_A_split_types,
			mtk::oztcecgemm::detail::matrix_A,
			&two_to_alpha,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_A");

	handle->profiler.start_timer_sync("split_B");
	split_core(
			reinterpret_cast<std::uint8_t*>(working_memory_ptr) + b_offset,
			op_B,
			k, n,
			b_ptr, ldb,
			split_config.matrix_B_split_types,
			mtk::oztcecgemm::detail::matrix_B,
			&two_to_alpha,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_B");
}

template <class T>
void split_AB_int8(
		mtk::oztcecgemm::handle_t handle,
		void* working_memory_ptr,
		const mtk::oztcecgemm::operation_t op_A,
		const mtk::oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const T* const a_ptr, const std::size_t lda,
		T* const a_max_exp_ptr,
		const T* const b_ptr, const std::size_t ldb,
		T* const b_max_exp_ptr,
		const unsigned num_split,
		const unsigned bits_per_int8
		) {
	const auto working_a_ptr = reinterpret_cast<std::int8_t*>(working_memory_ptr);

	handle->profiler.start_timer_sync("split_A");
	mtk::oztcecgemm::split_int8<T>(
			working_a_ptr,
			a_max_exp_ptr,
			m, k,
			a_ptr, lda,
			op_A,
			mtk::oztcecgemm::detail::matrix_A,
			num_split,
			bits_per_int8,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_A");

	const auto working_b_ptr = reinterpret_cast<std::int8_t*>(working_memory_ptr) + m * k * num_split;

	handle->profiler.start_timer_sync("split_B");
	mtk::oztcecgemm::split_int8<T>(
			working_b_ptr,
			b_max_exp_ptr,
			k, n,
			b_ptr, ldb,
			op_B,
			mtk::oztcecgemm::detail::matrix_B,
			num_split,
			bits_per_int8,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_B");
}

cudaDataType_t to_cudaDataType_t(
		const mtk::oztcecgemm::data_t d
		) {
	switch (d) {
	case mtk::oztcecgemm::fp32:
		return CUDA_R_32F;
	case mtk::oztcecgemm::fp16:
		return CUDA_R_16F;
	default:
		break;
	}
	OZTCECGEM_NOT_IMPLEMENTED;
	return CUDA_R_32F;
}

cublasOperation_t to_cublasOperation_t(
		const mtk::oztcecgemm::operation_t op
		) {
	switch (op) {
	case mtk::oztcecgemm::op_n:
		return CUBLAS_OP_N;
	case mtk::oztcecgemm::op_t:
		return CUBLAS_OP_T;
	default:
		break;
	}
	OZTCECGEM_NOT_IMPLEMENTED;
	return CUBLAS_OP_N;
}

mtk::shgemm::operation_t to_shgemm_operation_t(
		const mtk::oztcecgemm::operation_t op
		) {
	switch (op) {
	case mtk::oztcecgemm::op_n:
		return mtk::shgemm::op_n;
	case mtk::oztcecgemm::op_t:
		return mtk::shgemm::op_t;
	default:
		break;
	}
	OZTCECGEM_NOT_IMPLEMENTED;
	return mtk::shgemm::op_n;
}

__global__ void accumulate_in_fp64_kernel(
		double* const dp_ptr,
		const float* sp_ptr,
		const std::size_t length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}

	dp_ptr[tid] += sp_ptr[tid];
}

void accumulate_in_fp64(
		double* const dp_ptr,
		const float* sp_ptr,
		const std::size_t length,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	accumulate_in_fp64_kernel
		<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				dp_ptr,
				sp_ptr,
				length
			);
}

__global__ void accumulate_in_i64_kernel(
		std::int64_t* const i64_ptr,
		const std::int32_t* i32_ptr,
		const std::size_t length,
		const unsigned mantissa_rshift
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}

	i64_ptr[tid] += ((static_cast<std::int64_t>(i32_ptr[tid]) << 32) >> mantissa_rshift);
}

void accumulate_in_i64(
		std::int64_t* const i64_ptr,
		const std::int32_t* i32_ptr,
		const std::size_t length,
		const unsigned mantissa_rshift,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	accumulate_in_i64_kernel
		<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				i64_ptr,
				i32_ptr,
				length,
				mantissa_rshift
			);
}

template <class T>
__global__ void init_accumulator_buffer_kernel(
		T* const dp_ptr,
		const std::size_t length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}

	dp_ptr[tid] = 0;
}

template <class T>
void init_accumulator_buffer(
		T* const dp_ptr,
		const std::size_t length,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	init_accumulator_buffer_kernel<T>
		<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				dp_ptr,
				length
			);
}

template <class Y_T>
__global__ void axby_kernel(
		const std::size_t m,
		const std::size_t n,
		const double a,
		const double* const x_ptr,
		const double b,
		Y_T* const y_ptr,
		const std::size_t ldy
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto mi = tid % m;
	const auto ni = tid / m;

	const auto memory_index = ni * ldy + mi;

	if (b != 0) {
		y_ptr[memory_index] = a * x_ptr[tid] + b * y_ptr[memory_index];
	} else {
		y_ptr[memory_index] = a * x_ptr[tid];
	}
}

template <class Y_T>
void axby(
		const std::size_t m,
		const std::size_t n,
		const double a,
		const double* const x_ptr,
		const double b,
		Y_T* const y_ptr, const std::size_t ldy,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	axby_kernel
		<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				m, n,
				a,
				x_ptr,
				b,
				y_ptr, ldy
			);
}

__global__ void axby_kernel(
		const std::size_t m,
		const std::size_t n,
		const double a,
		const std::int64_t* const x_ptr,
		const double b,
		double* const y_ptr,
		const std::size_t ldy,
		const double* const a_max_exp_ptr,
		const double* const b_max_exp_ptr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto mi = tid % m;
	const auto ni = tid / m;

	const auto memory_index = ni * ldy + mi;

	const auto x = static_cast<double>(x_ptr[tid]) / (1lu << 44) * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

	if (b != 0) {
		y_ptr[memory_index] = a * x + b * y_ptr[memory_index];
	} else {
		y_ptr[memory_index] = a * x;
	}
}

void axby(
		const std::size_t m,
		const std::size_t n,
		const double a,
		const std::int64_t* const x_ptr,
		const double b,
		double* const y_ptr,
		const std::size_t ldy,
		const double* const a_max_exp_ptr,
		const double* const b_max_exp_ptr,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	axby_kernel
		<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				m, n,
				a,
				x_ptr,
				b,
				y_ptr, ldy,
				a_max_exp_ptr,
				b_max_exp_ptr
			);
}

void gemm_core(
		mtk::oztcecgemm::handle_t handle,
		const mtk::oztcecgemm::operation_t op_A,
		const mtk::oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* const a_ptr, const std::size_t lda, const mtk::oztcecgemm::data_t type_a,
		const void* const b_ptr, const std::size_t ldb, const mtk::oztcecgemm::data_t type_b,
		void* const c_ptr,
		const mtk::oztcecgemm::detail::gemm_pair_config_t& gemm_pair_config,
		const mtk::oztcecgemm::compute_mode_t compute_mode,
		void* const working_memory_ptr
		) {
	const auto gemm_mode = gemm_pair_config.gemm_mode;
	const auto split_config = mtk::oztcecgemm::detail::get_split_config(compute_mode);
	const auto lda_r = gemm_pair_config.A_id == 0 ? lda : k;
	const auto ldb_r = gemm_pair_config.B_id == 0 ? ldb : k;

	std::size_t A_working_ptr_offset = 0;
	for (unsigned i = 0; i < gemm_pair_config.A_id; i++) {
		const auto t = split_config.matrix_A_split_types[i];
		A_working_ptr_offset += m * k * mtk::oztcecgemm::get_data_size_in_byte(t);
	}

	std::size_t B_working_ptr_offset = 0;
	for (const auto t : split_config.matrix_A_split_types) {
		B_working_ptr_offset += m * k * mtk::oztcecgemm::get_data_size_in_byte(t);
	}
	for (unsigned i = 0; i < gemm_pair_config.B_id; i++) {
		const auto t = split_config.matrix_B_split_types[i];
		B_working_ptr_offset += k * n * mtk::oztcecgemm::get_data_size_in_byte(t);
	}

	void* const a_working_ptr = reinterpret_cast<std::uint8_t*>(working_memory_ptr) + A_working_ptr_offset;
	void* const b_working_ptr = reinterpret_cast<std::uint8_t*>(working_memory_ptr) + B_working_ptr_offset;

	const void* const a_ptr_r = gemm_pair_config.A_id == 0 ? a_ptr : a_working_ptr;
	const void* const b_ptr_r = gemm_pair_config.B_id == 0 ? b_ptr : b_working_ptr;
	void* const c_ptr_r = c_ptr;

	const float alpha_r = 1, beta_r = 0;

	const auto profile_label = mtk::oztcecgemm::detail::gemm_mode_str(gemm_mode);
	handle->profiler.start_timer_sync(profile_label);
	switch (gemm_mode) {
	case mtk::oztcecgemm::detail::cublas_dgemm:
		{
			const double alpha_dp = 1, beta_dp = 0;
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

			const auto cublas_algorithm = CUBLAS_GEMM_DEFAULT;

			auto cublas_compute_mode = CUBLAS_COMPUTE_64F;

			CUTF_CHECK_ERROR(cublasGemmEx(
						handle->cublas_handle,
						op_A_r,
						op_B_r,
						m, n, k,
						&alpha_dp,
						a_ptr_r, CUDA_R_64F, lda_r,
						b_ptr_r, CUDA_R_64F, ldb_r,
						&beta_dp,
						c_ptr_r, CUDA_R_64F, m,
						cublas_compute_mode,
						cublas_algorithm
						));
		}
		break;
	case mtk::oztcecgemm::detail::cublas_sgemm:
	case mtk::oztcecgemm::detail::cublas_bf16:
	case mtk::oztcecgemm::detail::cublas_tf32:
	case mtk::oztcecgemm::detail::cublas_fp16:
		{
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;
			const auto type_A_r = gemm_pair_config.A_id == 0 ? type_a : split_config.matrix_A_split_types[gemm_pair_config.A_id];
			const auto type_B_r = gemm_pair_config.B_id == 0 ? type_b : split_config.matrix_B_split_types[gemm_pair_config.B_id];

			const auto cublas_algorithm = gemm_mode == mtk::oztcecgemm::detail::cublas_sgemm ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

			auto cublas_compute_mode = CUBLAS_COMPUTE_32F;
			if (gemm_mode == mtk::oztcecgemm::detail::cublas_bf16) cublas_compute_mode = CUBLAS_COMPUTE_32F_FAST_16BF;
			else if (gemm_mode == mtk::oztcecgemm::detail::cublas_fp16) cublas_compute_mode = CUBLAS_COMPUTE_32F_FAST_16F;
			else if (gemm_mode == mtk::oztcecgemm::detail::cublas_tf32) cublas_compute_mode = CUBLAS_COMPUTE_32F_FAST_TF32;

			CUTF_CHECK_ERROR(cublasGemmEx(
						handle->cublas_handle,
						op_A_r,
						op_B_r,
						m, n, k,
						&alpha_r,
						a_ptr_r, to_cudaDataType_t(type_A_r), lda_r,
						b_ptr_r, to_cudaDataType_t(type_B_r), ldb_r,
						&beta_r,
						c_ptr_r, CUDA_R_32F, m,
						cublas_compute_mode,
						cublas_algorithm
						));
		}
		break;
	case mtk::oztcecgemm::detail::shgemm_tf32:
	case mtk::oztcecgemm::detail::shgemm_fp16:
		{
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_shgemm_operation_t(op_A) : mtk::shgemm::op_t;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_shgemm_operation_t(op_B) : mtk::shgemm::op_n;
			const auto shgemm_mode = gemm_mode == mtk::oztcecgemm::detail::shgemm_fp16 ? mtk::shgemm::fp16 : mtk::shgemm::tf32;
			mtk::shgemm::shgemm(
					handle->shgemm_handle,
					op_A_r,
					op_B_r,
					m, n, k,
					&alpha_r,
					reinterpret_cast<const float*>(a_ptr_r), lda_r,
					reinterpret_cast<const half*>(b_ptr_r), ldb_r,
					&beta_r,
					reinterpret_cast<float*>(c_ptr_r), m,
					shgemm_mode
					);
		}
		break;
	case mtk::oztcecgemm::detail::hsgemm_tf32:
	case mtk::oztcecgemm::detail::hsgemm_fp16:
		{
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_shgemm_operation_t(op_A) : mtk::shgemm::op_t;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_shgemm_operation_t(op_B) : mtk::shgemm::op_n;
			const auto shgemm_mode = gemm_mode == mtk::oztcecgemm::detail::hsgemm_fp16 ? mtk::shgemm::fp16 : mtk::shgemm::tf32;
			mtk::shgemm::hsgemm(
					handle->shgemm_handle,
					op_A_r,
					op_B_r,
					m, n, k,
					&alpha_r,
					reinterpret_cast<const half*>(a_ptr_r), lda_r,
					reinterpret_cast<const float*>(b_ptr_r), ldb_r,
					&beta_r,
					reinterpret_cast<float*>(c_ptr_r), m,
					shgemm_mode
					);
		}
		break;
	case mtk::oztcecgemm::detail::fp16tcec:
	case mtk::oztcecgemm::detail::tf32tcec:
		{
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;
			const auto type_A_r = gemm_pair_config.A_id == 0 ? type_a : split_config.matrix_A_split_types[gemm_pair_config.A_id];
			const auto type_B_r = gemm_pair_config.B_id == 0 ? type_b : split_config.matrix_B_split_types[gemm_pair_config.B_id];

			const auto cumpsgemm_compute_mode = gemm_mode == mtk::oztcecgemm::detail::fp16tcec ? CUMPSGEMM_FP16TCEC : CUMPSGEMM_TF32TCEC;

			cumpsgemm::gemm(
						handle->cumpsgemm_handle,
						op_A_r,
						op_B_r,
						m, n, k,
						&alpha_r,
						reinterpret_cast<const float*>(a_ptr_r), lda_r,
						reinterpret_cast<const float*>(b_ptr_r), ldb_r,
						&beta_r,
						reinterpret_cast<float*>(c_ptr_r), m,
						cumpsgemm_compute_mode
						);
		}
		break;
	case mtk::oztcecgemm::detail::int8tc:
		{
			const int alpha_i = 1, beta_i = 0;
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

			CUTF_CHECK_ERROR(cublasGemmEx(
						handle->cublas_handle,
						op_A_r,
						op_B_r,
						m, n, k,
						&alpha_i,
						a_ptr_r, CUDA_R_8I, lda_r,
						b_ptr_r, CUDA_R_8I, ldb_r,
						&beta_i,
						c_ptr_r, CUDA_R_32I, m,
						CUBLAS_COMPUTE_32I,
						CUBLAS_GEMM_DEFAULT_TENSOR_OP
						));
		}
		break;
	default:
		OZTCECGEM_NOT_IMPLEMENTED;
	}
	handle->profiler.stop_timer_sync(profile_label);
}
} // unnamed namespace

int mtk::oztcecgemm::gemm(
		mtk::oztcecgemm::handle_t handle,
		const mtk::oztcecgemm::operation_t op_A,
		const mtk::oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* alpha,
		const void* const a_ptr, const std::size_t lda,
		const void* const b_ptr, const std::size_t ldb,
		const void* beta,
		void* const c_ptr, std::size_t ldc,
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	mtk::oztcecgemm::data_t input_type;
	switch (compute_mode) {
	case mtk::oztcecgemm::fp32_split_3:
	case mtk::oztcecgemm::sgemm:
		input_type = mtk::oztcecgemm::fp32;
		break;
	case mtk::oztcecgemm::dgemm:
	case mtk::oztcecgemm::fp64_int8_6:
	case mtk::oztcecgemm::fp64_int8_7:
	case mtk::oztcecgemm::fp64_int8_8:
	case mtk::oztcecgemm::fp64_int8_9:
		input_type = mtk::oztcecgemm::fp64;
		break;
	default:
		OZTCECGEM_NOT_IMPLEMENTED;
	}

	if (input_type == mtk::oztcecgemm::fp32) {
		float*  const c_fp32_ptr = reinterpret_cast<float* >(handle->working_memory_ptr);
		double* const c_fp64_ptr = reinterpret_cast<double*>(c_fp32_ptr + m * n);
		void*   const working_memory_ptr = c_fp64_ptr + m * n;

		init_accumulator_buffer<double>(c_fp64_ptr, m * n, handle->cuda_stream);

		split_AB(
				handle,
				working_memory_ptr,
				op_A, op_B,
				m, n, k,
				reinterpret_cast<const float*>(a_ptr), lda,
				reinterpret_cast<const float*>(b_ptr), ldb,
				compute_mode
				);
		const auto& gemm_pair_config_list = mtk::oztcecgemm::detail::get_split_config(compute_mode).gemm_pair_config_list;
		for (const auto& gemm_pair_config : gemm_pair_config_list) {
			gemm_core(
					handle,
					op_A, op_B,
					m, n, k,
					a_ptr, lda, input_type,
					b_ptr, ldb, input_type,
					c_fp32_ptr,
					gemm_pair_config,
					compute_mode,
					working_memory_ptr
					);
			handle->profiler.start_timer_sync("accumulate_in_fp64");
			accumulate_in_fp64(c_fp64_ptr, c_fp32_ptr, m * n, handle->cuda_stream);
			handle->profiler.stop_timer_sync("accumulate_in_fp64");
		}

		handle->profiler.start_timer_sync("copy_result");
		if (mtk::oztcecgemm::get_output_type(compute_mode) == fp32) {
			using C_T = float;
			axby<C_T>(
					m, n,
					*reinterpret_cast<const C_T*>(alpha),
					c_fp64_ptr,
					*reinterpret_cast<const C_T*>(beta),
					reinterpret_cast<C_T*>(c_ptr), ldc,
					handle->cuda_stream
					);
		} else {
			using C_T = double;
			axby<C_T>(
					m, n,
					*reinterpret_cast<const C_T*>(alpha),
					c_fp64_ptr,
					*reinterpret_cast<const C_T*>(beta),
					reinterpret_cast<C_T*>(c_ptr), ldc,
					handle->cuda_stream
					);
		}
		handle->profiler.stop_timer_sync("copy_result");
	} else if (input_type == mtk::oztcecgemm::fp64) {
		if (
				compute_mode == mtk::oztcecgemm::fp64_int8_6 ||
				compute_mode == mtk::oztcecgemm::fp64_int8_7 ||
				compute_mode == mtk::oztcecgemm::fp64_int8_8 ||
				compute_mode == mtk::oztcecgemm::fp64_int8_9) {
			const unsigned num_split = mtk::oztcecgemm::detail::get_split_config(compute_mode).matrix_A_split_types.size() - 1;
			const auto bits_per_int8 = std::min<unsigned>(7u, std::ceil((31 - std::log2(k) / 2.)));

			std::int32_t* const c_i32_ptr = reinterpret_cast<std::int32_t*>(handle->working_memory_ptr);
			std::int64_t* const c_i64_ptr = reinterpret_cast<std::int64_t*>(c_i32_ptr + m * n);
			double* const a_max_exp_ptr = reinterpret_cast<double*>(c_i64_ptr + m * n);
			double* const b_max_exp_ptr = a_max_exp_ptr + m;
			void* const working_memory_ptr = b_max_exp_ptr + n;

			init_accumulator_buffer<std::int64_t>(
					c_i64_ptr,
					m * n,
					handle->cuda_stream
					);

			split_AB_int8<double>(
					handle,
					working_memory_ptr,
					op_A,
					op_B,
					m, n, k,
					reinterpret_cast<const double*>(a_ptr), lda,
					a_max_exp_ptr,
					reinterpret_cast<const double*>(b_ptr), ldb,
					b_max_exp_ptr,
					num_split,
					bits_per_int8
					);

			const auto& gemm_pair_config_list = mtk::oztcecgemm::detail::get_split_config(compute_mode).gemm_pair_config_list;
			for (const auto& gemm_pair_config : gemm_pair_config_list) {
				gemm_core(
						handle,
						op_A, op_B,
						m, n, k,
						a_ptr, lda, input_type,
						b_ptr, ldb, input_type,
						c_i32_ptr,
						gemm_pair_config,
						compute_mode,
						working_memory_ptr
						);
				handle->profiler.start_timer_sync("accumulate_in_i64");
				accumulate_in_i64(
						c_i64_ptr,
						c_i32_ptr,
						m * n,
						bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2),
						handle->cuda_stream
						);
				handle->profiler.stop_timer_sync("accumulate_in_i64");
			}
			using C_T = double;
			handle->profiler.start_timer_sync("copy_result");
			axby(
					m, n,
					*reinterpret_cast<const C_T*>(alpha),
					c_i64_ptr,
					*reinterpret_cast<const C_T*>(beta),
					reinterpret_cast<C_T*>(c_ptr), ldc,
					a_max_exp_ptr,
					b_max_exp_ptr,
					handle->cuda_stream
					);
			handle->profiler.stop_timer_sync("copy_result");
		} else if (compute_mode == mtk::oztcecgemm::dgemm) {
				const auto& gemm_pair_config_list = mtk::oztcecgemm::detail::get_split_config(compute_mode).gemm_pair_config_list;
				gemm_core(
						handle,
						op_A, op_B,
						m, n, k,
						a_ptr, lda, input_type,
						b_ptr, ldb, input_type,
						c_ptr,
						gemm_pair_config_list[0],
						compute_mode,
						nullptr
						);
		} else {
			OZTCECGEM_NOT_IMPLEMENTED;
		}
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}
	return 0;
}
