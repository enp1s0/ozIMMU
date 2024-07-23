#include <cutf/cublas.hpp>
#include "config.hpp"
#include "split.hpp"
#include "utils.hpp"
#include "handle.hpp"

namespace {
template <class T>
std::size_t split_core(
		void* const split_ptr,
		const mtk::ozimmu::operation_t op,
		const std::size_t m,
		const std::size_t n,
		const T* const src_ptr, const std::size_t ld,
		const std::vector<mtk::ozimmu::data_t> data_type_list,
		const mtk::ozimmu::detail::matrix_t matrix,
		const T* const two_to_alpha_ptr,
		cudaStream_t cuda_stream
		) {
	const auto num_split = data_type_list.size() - 1;
	std::size_t offset = 0;

	if (num_split <= 1) {
		// Do nothing
	} else {
		OZIMMU_NOT_IMPLEMENTED;
	}

	return offset;
}

template <class T>
void split_AB_int8(
		mtk::ozimmu::handle_t handle,
		const mtk::ozimmu::operation_t op_A,
		const mtk::ozimmu::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const T* const a_ptr, const std::size_t lda,
		double* const a_max_exp_ptr,
		std::int8_t* const working_a_ptr,
		const T* const b_ptr, const std::size_t ldb,
		double* const b_max_exp_ptr,
		std::int8_t* const working_b_ptr,
		const unsigned num_split,
		const unsigned bits_per_int8
		) {
	handle->profiler.start_timer_sync("split_A");
	mtk::ozimmu::split_int8<T>(
			working_a_ptr,
			a_max_exp_ptr,
			m, k,
			a_ptr, lda,
			op_A,
			mtk::ozimmu::detail::matrix_A,
			num_split,
			bits_per_int8,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_A");

	handle->profiler.start_timer_sync("split_B");
	mtk::ozimmu::split_int8<T>(
			working_b_ptr,
			b_max_exp_ptr,
			k, n,
			b_ptr, ldb,
			op_B,
			mtk::ozimmu::detail::matrix_B,
			num_split,
			bits_per_int8,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("split_B");
}

cudaDataType_t to_cudaDataType_t(
		const mtk::ozimmu::data_t d
		) {
	switch (d) {
	case mtk::ozimmu::fp32:
		return CUDA_R_32F;
	case mtk::ozimmu::fp16:
		return CUDA_R_16F;
	default:
		break;
	}
	OZIMMU_NOT_IMPLEMENTED;
	return CUDA_R_32F;
}

cublasOperation_t to_cublasOperation_t(
		const mtk::ozimmu::operation_t op
		) {
	switch (op) {
	case mtk::ozimmu::op_n:
		return CUBLAS_OP_N;
	case mtk::ozimmu::op_t:
		return CUBLAS_OP_T;
	default:
		break;
	}
	OZIMMU_NOT_IMPLEMENTED;
	return CUBLAS_OP_N;
}

__global__ void accumulate_in_f64_kernel(
		double* const f64_ptr,
		const std::int32_t* i32_ptr,
		const std::size_t length,
		const double scale
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}

	f64_ptr[tid] += static_cast<double>(static_cast<std::int64_t>(i32_ptr[tid]) << 32) * scale;
}

void accumulate_in_f64(
		double* const f64_ptr,
		const std::int32_t* i32_ptr,
		const std::size_t length,
		const std::int32_t mantissa_rshift,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	const auto scale = cutf::experimental::fp::reinterpret_as_fp(static_cast<std::uint64_t>((cutf::experimental::fp::get_bias<double>() - mantissa_rshift)) << cutf::experimental::fp::get_mantissa_size<double>());
	accumulate_in_f64_kernel
		<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				f64_ptr,
				i32_ptr,
				length,
				scale
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

__global__ void axby_kernel(
		const std::size_t m,
		const std::size_t n,
		const double a,
		const double* const x_ptr,
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

	const auto x = x_ptr[tid] / (1l << 44) * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

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
		const double* const x_ptr,
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

__global__ void axy_complex_kernel(
		const std::size_t m,
		const std::size_t n,
		const cuDoubleComplex a,
		const double* const x_ptr,
		cuDoubleComplex* const y_ptr,
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

	const auto x = x_ptr[tid] / (1l << 44) * a_max_exp_ptr[mi] * b_max_exp_ptr[ni];

	auto y = y_ptr[memory_index];

	y.x = a.x * x + y.x;
	y.y = a.y * x + y.y;

	y_ptr[memory_index] = y;
}


void axy_complex(
		const std::size_t m,
		const std::size_t n,
		const cuDoubleComplex a,
		const double* const x_ptr,
		cuDoubleComplex* const y_ptr,
		const std::size_t ldy,
		const double* const a_max_exp_ptr,
		const double* const b_max_exp_ptr,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;
	axy_complex_kernel
		<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
				m, n,
				a,
				x_ptr,
				y_ptr, ldy,
				a_max_exp_ptr,
				b_max_exp_ptr
			);
}

template <bool is_beta_zero>
__global__ void init_c_complex_kernel(
		const std::size_t m,
		const std::size_t n,
		cuDoubleComplex* const c_ptr,
		const std::size_t ldc,
		const cuDoubleComplex beta
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto mi = tid % m;
	const auto ni = tid / m;

	const auto memory_index = ni * ldc + mi;

	if (is_beta_zero) {
		c_ptr[memory_index] = make_cuDoubleComplex(0, 0);
	} else {
		auto c = c_ptr[memory_index];
		c.x = c.x * beta.x - c.y * beta.y;
		c.y = c.y * beta.x + c.x * beta.y;

		c_ptr[memory_index] = c;
	}
}

void init_c_complex(
		const std::size_t m,
		const std::size_t n,
		cuDoubleComplex* const c_ptr,
		const std::size_t ldc,
		const cuDoubleComplex beta,
		cudaStream_t cuda_stream
		) {
	constexpr std::size_t block_size = 256;

	if (beta.x == 0 && beta.y == 0) {
		init_c_complex_kernel<true>
			<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
					m, n,
					c_ptr,
					ldc,
					beta
					);
	} else {
		init_c_complex_kernel<false>
			<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
					m, n,
					c_ptr,
					ldc,
					beta
					);
	}
}

cublasStatus_t cublasGemmEx_org(cublasHandle_t handle, cublasOperation_t transa,
		cublasOperation_t transb, int m, int n, int k,
		const void *alpha, const void *A,
		cudaDataType_t Atype, int lda, const void *B,
		cudaDataType_t Btype, int ldb, const void *beta,
		void *C, cudaDataType_t Ctype, int ldc,
		cublasComputeType_t computeType,
		cublasGemmAlgo_t algo) {
	const std::string cublas_library_name = "libcublas.so";
	const std::string cublas_function_name = "cublasGemmEx";
	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = ozIMMU_get_function_pointer(
			cublas_library_name.c_str(),
			cublas_function_name.c_str()
			);

	const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

	return res;
}

void gemm_core(
		mtk::ozimmu::handle_t handle,
		const mtk::ozimmu::operation_t op_A,
		const mtk::ozimmu::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* const a_ptr, const std::size_t lda, const mtk::ozimmu::data_t type_a,
		const void* const b_ptr, const std::size_t ldb, const mtk::ozimmu::data_t type_b,
		void* const c_ptr,
		const mtk::ozimmu::detail::gemm_pair_config_t& gemm_pair_config,
		const mtk::ozimmu::compute_mode_t compute_mode,
		const void* const a_working_memory_ptr,
		const void* const b_working_memory_ptr
		) {
	const auto gemm_mode = gemm_pair_config.gemm_mode;
	const auto split_config = mtk::ozimmu::detail::get_split_config(compute_mode);
	const auto lda_r = gemm_pair_config.A_id == 0 ? lda : k;
	const auto ldb_r = gemm_pair_config.B_id == 0 ? ldb : k;

	std::size_t A_working_ptr_offset = 0;
	for (unsigned i = 0; i < gemm_pair_config.A_id; i++) {
		const auto t = split_config.matrix_A_split_types[i];
		A_working_ptr_offset += m * k * mtk::ozimmu::get_data_size_in_byte(t);
	}

	std::size_t B_working_ptr_offset = 0;
	for (unsigned i = 0; i < gemm_pair_config.B_id; i++) {
		const auto t = split_config.matrix_B_split_types[i];
		B_working_ptr_offset += k * n * mtk::ozimmu::get_data_size_in_byte(t);
	}

	const void* const a_working_ptr = reinterpret_cast<const std::uint8_t*>(a_working_memory_ptr) + A_working_ptr_offset;
	const void* const b_working_ptr = reinterpret_cast<const std::uint8_t*>(b_working_memory_ptr) + B_working_ptr_offset;

	const void* const a_ptr_r = gemm_pair_config.A_id == 0 ? a_ptr : a_working_ptr;
	const void* const b_ptr_r = gemm_pair_config.B_id == 0 ? b_ptr : b_working_ptr;
	void* const c_ptr_r = c_ptr;

	const float alpha_r = 1, beta_r = 0;

	const auto profile_label = mtk::ozimmu::detail::gemm_mode_str(gemm_mode);
	handle->profiler.start_timer_sync(profile_label);
	switch (gemm_mode) {
	case mtk::ozimmu::detail::cublas_dgemm:
		{
			const double alpha_dp = 1, beta_dp = 0;
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

			const auto cublas_algorithm = CUBLAS_GEMM_DEFAULT;

			const auto cublas_compute_mode = CUBLAS_COMPUTE_64F;

			CUTF_CHECK_ERROR(cublasGemmEx_org(
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
	case mtk::ozimmu::detail::cublas_sgemm:
	case mtk::ozimmu::detail::cublas_bf16:
	case mtk::ozimmu::detail::cublas_tf32:
	case mtk::ozimmu::detail::cublas_fp16:
		{
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;
			const auto type_A_r = gemm_pair_config.A_id == 0 ? type_a : split_config.matrix_A_split_types[gemm_pair_config.A_id];
			const auto type_B_r = gemm_pair_config.B_id == 0 ? type_b : split_config.matrix_B_split_types[gemm_pair_config.B_id];

			const auto cublas_algorithm = gemm_mode == mtk::ozimmu::detail::cublas_sgemm ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

			auto cublas_compute_mode = CUBLAS_COMPUTE_32F;
			if (gemm_mode == mtk::ozimmu::detail::cublas_bf16) cublas_compute_mode = CUBLAS_COMPUTE_32F_FAST_16BF;
			else if (gemm_mode == mtk::ozimmu::detail::cublas_fp16) cublas_compute_mode = CUBLAS_COMPUTE_32F_FAST_16F;
			else if (gemm_mode == mtk::ozimmu::detail::cublas_tf32) cublas_compute_mode = CUBLAS_COMPUTE_32F_FAST_TF32;

			CUTF_CHECK_ERROR(cublasGemmEx_org(
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
	case mtk::ozimmu::detail::int8tc:
		{
			const int alpha_i = 1, beta_i = 0;
			const auto op_A_r = gemm_pair_config.A_id == 0 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == 0 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

			CUTF_CHECK_ERROR_M(cublasGemmEx_org(
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
						), ("GemmEx(int8)-m" + std::to_string(m) + "-n" + std::to_string(n) + "-k" + std::to_string(k)));
		}
		break;
	default:
		OZIMMU_NOT_IMPLEMENTED;
	}
	handle->profiler.stop_timer_sync(profile_label);
}

template <class T>
int gemm_int8(
		mtk::ozimmu::handle_t handle,
		const mtk::ozimmu::operation_t op_A,
		const mtk::ozimmu::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const T* alpha,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		const T* beta,
		T* const c_ptr, std::size_t ldc,
		const mtk::ozimmu::compute_mode_t compute_mode
		);

template <>
int gemm_int8<double>(
		mtk::ozimmu::handle_t handle,
		const mtk::ozimmu::operation_t op_A,
		const mtk::ozimmu::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const double* alpha,
		const double* const a_ptr, const std::size_t lda,
		const double* const b_ptr, const std::size_t ldb,
		const double* beta,
		double* const c_ptr, std::size_t ldc,
		const mtk::ozimmu::compute_mode_t compute_mode
		) {
	const unsigned num_split = mtk::ozimmu::detail::get_split_config(compute_mode).matrix_A_split_types.size() - 1;
	const int32_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);

	std::int32_t* const c_i32_ptr = reinterpret_cast<std::int32_t*>(handle->working_memory_ptr);
	double* const c_f64_ptr = reinterpret_cast<double*>(c_i32_ptr + m * n);
	double* const a_max_exp_ptr = reinterpret_cast<double*>(c_f64_ptr + m * n);
	double* const b_max_exp_ptr = a_max_exp_ptr + m;
	void* const working_memory_ptr = b_max_exp_ptr + n;

	init_accumulator_buffer(
			c_f64_ptr,
			m * n,
			handle->cuda_stream
			);

	split_AB_int8<double>(
			handle,
			op_A,
			op_B,
			m, n, k, a_ptr, lda,
			a_max_exp_ptr,
			reinterpret_cast<std::int8_t*>(working_memory_ptr),
			b_ptr, ldb,
			b_max_exp_ptr,
			reinterpret_cast<std::int8_t*>(working_memory_ptr) + m * k * num_split,
			num_split,
			bits_per_int8
			);

	std::size_t A_working_memory_size = mtk::ozimmu::detail::calculate_working_memory_size(m, k, compute_mode, mtk::ozimmu::detail::matrix_A, mtk::ozimmu::real);

	const auto& gemm_pair_config_list = mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;
	for (const auto& gemm_pair_config : gemm_pair_config_list) {
		gemm_core(
				handle,
				op_A, op_B,
				m, n, k,
				a_ptr, lda, mtk::ozimmu::fp64,
				b_ptr, ldb, mtk::ozimmu::fp64,
				c_i32_ptr,
				gemm_pair_config,
				compute_mode,
				working_memory_ptr,
				reinterpret_cast<std::uint8_t*>(working_memory_ptr) + A_working_memory_size
				);
		handle->profiler.start_timer_sync("accumulate_in_f64");
		accumulate_in_f64(
				c_f64_ptr,
				c_i32_ptr,
				m * n,
				bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2) - (7 /*bitlen(int8)-1*/ - bits_per_int8) * 2, // The `(7 - bits_per_int8) * 2` term is required because the mantissa `bits_per_int8` bits are stored in the low `bits_per_int8` bits of an int8.
				handle->cuda_stream
				);
		handle->profiler.stop_timer_sync("accumulate_in_f64");
	}
	handle->profiler.start_timer_sync("copy_result");
	axby(
			m, n,
			*alpha,
			c_f64_ptr,
			*beta,
			c_ptr, ldc,
			a_max_exp_ptr,
			b_max_exp_ptr,
			handle->cuda_stream
			);
	handle->profiler.stop_timer_sync("copy_result");

	return 0;
}

template <>
int gemm_int8<cuDoubleComplex>(
		mtk::ozimmu::handle_t handle,
		const mtk::ozimmu::operation_t op_A,
		const mtk::ozimmu::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const cuDoubleComplex* alpha,
		const cuDoubleComplex* const a_ptr, const std::size_t lda,
		const cuDoubleComplex* const b_ptr, const std::size_t ldb,
		const cuDoubleComplex* beta,
		cuDoubleComplex* const c_ptr, std::size_t ldc,
		const mtk::ozimmu::compute_mode_t compute_mode
		) {
	using real_t = double;
	const unsigned num_split = mtk::ozimmu::detail::get_split_config(compute_mode).matrix_A_split_types.size() - 1;
	const int32_t bits_per_int8 = mtk::ozimmu::get_bits_per_int8(k);
	const auto& gemm_pair_config_list = mtk::ozimmu::detail::get_split_config(compute_mode).gemm_pair_config_list;

	std::int32_t* const c_i32_ptr = reinterpret_cast<std::int32_t*>(handle->working_memory_ptr);
	double* const tmp_f64_ptr = reinterpret_cast<double*>(c_i32_ptr + m * n);
	double* const a_real_max_exp_ptr = reinterpret_cast<double*>(tmp_f64_ptr + m * n);
	double* const a_imag_max_exp_ptr = a_real_max_exp_ptr + m;
	double* const b_real_max_exp_ptr = a_imag_max_exp_ptr + m;
	double* const b_imag_max_exp_ptr = b_real_max_exp_ptr + n;
	void* const working_memory_ptr = b_imag_max_exp_ptr + n;

	const double* a_max_exp_ptr_list[] = {
		a_real_max_exp_ptr,
		a_imag_max_exp_ptr
	};
	const std::int8_t* a_int8_working_memory_ptr_list[] = {
		reinterpret_cast<const std::int8_t*>(working_memory_ptr),
		reinterpret_cast<const std::int8_t*>(working_memory_ptr) + mtk::ozimmu::detail::calculate_working_memory_size(m, k, compute_mode, mtk::ozimmu::detail::matrix_A, mtk::ozimmu::real),
	};

	const double* b_max_exp_ptr_list[] = {
		b_real_max_exp_ptr,
		b_imag_max_exp_ptr
	};
	const std::int8_t* b_int8_working_memory_ptr_list[] = {
		a_int8_working_memory_ptr_list[0] + mtk::ozimmu::detail::calculate_working_memory_size(m, k, compute_mode, mtk::ozimmu::detail::matrix_A, mtk::ozimmu::complx),
		a_int8_working_memory_ptr_list[0] + mtk::ozimmu::detail::calculate_working_memory_size(m, k, compute_mode, mtk::ozimmu::detail::matrix_A, mtk::ozimmu::complx) + mtk::ozimmu::detail::calculate_working_memory_size(k, n, compute_mode, mtk::ozimmu::detail::matrix_B, mtk::ozimmu::real),
	};

	split_AB_int8<cuDoubleComplex>(
			handle,
			op_A,
			op_B,
			m, n, k,
			a_ptr, lda,
			a_real_max_exp_ptr,
			reinterpret_cast<std::int8_t*>(working_memory_ptr),
			b_ptr, ldb,
			b_real_max_exp_ptr,
			reinterpret_cast<std::int8_t*>(working_memory_ptr) + m * k * num_split * 2,
			num_split,
			bits_per_int8
			);

	// Init C
	init_c_complex(
			m, n,
			c_ptr, ldc,
			*beta,
			handle->cuda_stream
			);

	for (const auto p : std::vector<std::pair<unsigned, unsigned>>{{1, 1}, {0, 0}, {1, 0}, {0, 1}}) {
		init_accumulator_buffer(
				tmp_f64_ptr,
				m * n,
				handle->cuda_stream
				);
		for (const auto& gemm_pair_config : gemm_pair_config_list) {
			gemm_core(
					handle,
					op_A, op_B,
					m, n, k,
					a_ptr, lda, mtk::ozimmu::fp64,
					b_ptr, ldb, mtk::ozimmu::fp64,
					c_i32_ptr,
					gemm_pair_config,
					compute_mode,
					a_int8_working_memory_ptr_list[p.first],
					b_int8_working_memory_ptr_list[p.second]
					);
			handle->profiler.start_timer_sync("accumulate_in_f64");
			accumulate_in_f64(
					tmp_f64_ptr,
					c_i32_ptr,
					m * n,
					bits_per_int8 * (gemm_pair_config.A_id + gemm_pair_config.B_id - 2) - (7 /*bitlen(int8)-1*/ - bits_per_int8) * 2, // The `(7 - bits_per_int8) * 2` term is required because the mantissa `bits_per_int8` bits are stored in the low `bits_per_int8` bits of an int8.
					handle->cuda_stream
					);
			handle->profiler.stop_timer_sync("accumulate_in_f64");
		}

		real_t axpy_alpha_real = 0;
		real_t axpy_alpha_imag = 0;
		if (p.first == 0 && p.second == 0) {
			axpy_alpha_real = alpha->x;
			axpy_alpha_imag = alpha->y;
		} else if (p.first == 1 && p.second == 1) {
			axpy_alpha_real = -alpha->x;
			axpy_alpha_imag = -alpha->y;
		} else {
			axpy_alpha_real = -alpha->y;
			axpy_alpha_imag = alpha->x;
		}
		handle->profiler.start_timer_sync("copy_result");
		axy_complex(
				m, n,
				make_cuDoubleComplex(axpy_alpha_real, axpy_alpha_imag),
				tmp_f64_ptr,
				c_ptr, ldc,
				a_max_exp_ptr_list[p.first],
				b_max_exp_ptr_list[p.second],
				handle->cuda_stream
				);
		handle->profiler.stop_timer_sync("copy_result");
	}

	return 0;
}
} // unnamed namespace

int mtk::ozimmu::gemm(
		mtk::ozimmu::handle_t handle,
		const mtk::ozimmu::operation_t op_A,
		const mtk::ozimmu::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* alpha,
		const void* const a_ptr, const std::size_t lda,
		const void* const b_ptr, const std::size_t ldb,
		const void* beta,
		void* const c_ptr, std::size_t ldc,
		const mtk::ozimmu::compute_mode_t compute_mode,
		const mtk::ozimmu::element_kind_t element_kind
		) {
	// Arguments validation
	int arg_errors = 0;
	arg_errors += check_gemm_shape(op_A, m, k, lda, "A");
	arg_errors += check_gemm_shape(op_B, k, n, ldb, "B");
	arg_errors += check_gemm_shape(mtk::ozimmu::op_n, m, n, ldc, "C");
	if (arg_errors) {
		return 1;
	}

	mtk::ozimmu::data_t input_type;
	switch (compute_mode) {
		case mtk::ozimmu::sgemm:
			input_type = mtk::ozimmu::fp32;
			break;
		case mtk::ozimmu::dgemm:
		case mtk::ozimmu::fp64_int8_3:
		case mtk::ozimmu::fp64_int8_4:
		case mtk::ozimmu::fp64_int8_5:
		case mtk::ozimmu::fp64_int8_6:
		case mtk::ozimmu::fp64_int8_7:
		case mtk::ozimmu::fp64_int8_8:
		case mtk::ozimmu::fp64_int8_9:
		case mtk::ozimmu::fp64_int8_10:
		case mtk::ozimmu::fp64_int8_11:
		case mtk::ozimmu::fp64_int8_12:
		case mtk::ozimmu::fp64_int8_13:
		case mtk::ozimmu::fp64_int8_14:
		case mtk::ozimmu::fp64_int8_15:
		case mtk::ozimmu::fp64_int8_16:
		case mtk::ozimmu::fp64_int8_17:
		case mtk::ozimmu::fp64_int8_18:
		case mtk::ozimmu::fp64_int8_auto:
			input_type = mtk::ozimmu::fp64;
			break;
		default:
			OZIMMU_NOT_IMPLEMENTED;
	}

	gemm_list_t gemm_list = {
		std::tuple<std::size_t, std::size_t, std::size_t, mtk::ozimmu::element_kind_t, mtk::ozimmu::compute_mode_t>{m, n, k, element_kind, compute_mode}
	};
	mtk::ozimmu::reallocate_working_memory(handle, gemm_list);

	if (input_type == mtk::ozimmu::fp64) {
		if (
				compute_mode == mtk::ozimmu::fp64_int8_3  ||
				compute_mode == mtk::ozimmu::fp64_int8_4  ||
				compute_mode == mtk::ozimmu::fp64_int8_5  ||
				compute_mode == mtk::ozimmu::fp64_int8_6  ||
				compute_mode == mtk::ozimmu::fp64_int8_7  ||
				compute_mode == mtk::ozimmu::fp64_int8_8  ||
				compute_mode == mtk::ozimmu::fp64_int8_9  ||
				compute_mode == mtk::ozimmu::fp64_int8_10 ||
				compute_mode == mtk::ozimmu::fp64_int8_11 ||
				compute_mode == mtk::ozimmu::fp64_int8_12 ||
				compute_mode == mtk::ozimmu::fp64_int8_13 ||
				compute_mode == mtk::ozimmu::fp64_int8_14 ||
				compute_mode == mtk::ozimmu::fp64_int8_15 ||
				compute_mode == mtk::ozimmu::fp64_int8_16 ||
				compute_mode == mtk::ozimmu::fp64_int8_17 ||
				compute_mode == mtk::ozimmu::fp64_int8_18
				) {
			if (element_kind == mtk::ozimmu::real) {
				using T = double;
				gemm_int8(handle, op_A, op_B, m, n, k, reinterpret_cast<const T*>(alpha), reinterpret_cast<const T*>(a_ptr), lda, reinterpret_cast<const T*>(b_ptr), ldb, reinterpret_cast<const T*>(beta), reinterpret_cast<T*>(c_ptr), ldc, compute_mode);
			} else {
				using T = cuDoubleComplex;
				gemm_int8(handle, op_A, op_B, m, n, k, reinterpret_cast<const T*>(alpha), reinterpret_cast<const T*>(a_ptr), lda, reinterpret_cast<const T*>(b_ptr), ldb, reinterpret_cast<const T*>(beta), reinterpret_cast<T*>(c_ptr), ldc, compute_mode);
			}
		} else if (compute_mode == mtk::ozimmu::fp64_int8_auto) {
			const auto auto_mode = mtk::ozimmu::auto_mode_select(
						handle,
						op_A,
						op_B,
						m, n, k,
						a_ptr, lda,
						b_ptr, ldb,
						element_kind,
						handle->avg_mantissa_loss_threshold
					);
			ozIMMU_log("AUTO selected mode = " + mtk::ozimmu::get_compute_mode_name_str(auto_mode) + ", threshold average mantissa loss = " + std::to_string(handle->avg_mantissa_loss_threshold));
			return mtk::ozimmu::gemm(
					handle,
					op_A, op_B,
					m, n, k,
					alpha,
					a_ptr, lda,
					b_ptr, ldb,
					beta,
					c_ptr, ldc,
					auto_mode,
					element_kind
					);
		} else if (compute_mode == mtk::ozimmu::dgemm) {
			const auto dtype = element_kind == mtk::ozimmu::real ? CUDA_R_64F : CUDA_C_64F;
				cublasGemmEx_org(
						handle->cublas_handle,
						to_cublasOperation_t(op_A),
						to_cublasOperation_t(op_B),
						m, n, k,
						alpha,
						a_ptr, dtype, lda,
						b_ptr, dtype, ldb,
						beta,
						c_ptr, dtype, ldc,
						CUBLAS_COMPUTE_64F,
						CUBLAS_GEMM_DEFAULT
						);
		} else {
			OZIMMU_NOT_IMPLEMENTED;
		}
	} else {
		OZIMMU_NOT_IMPLEMENTED;
	}
	return 0;
}
