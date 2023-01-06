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
		const std::vector<mtk::oztcecgemm::detail::data_t> data_type_list,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		const T* const two_to_alpha_ptr,
		cudaStream_t cuda_stream
		) {
	std::size_t offset = 0;

	if (data_type_list.size() == 2) {
		mtk::oztcecgemm::split_2(
				reinterpret_cast<std::uint8_t*>(split_ptr), data_type_list[0],
				reinterpret_cast<std::uint8_t*>(split_ptr) + mtk::oztcecgemm::detail::get_data_size_in_byte(data_type_list[0]) * m * n, data_type_list[1],
				m, n,
				src_ptr, mtk::oztcecgemm::detail::get_data_t<T>(), ld,
				op,
				matrix,
				two_to_alpha_ptr,
				cuda_stream
				);
		offset += mtk::oztcecgemm::detail::get_data_size_in_byte(data_type_list[0]) * m * n;
		offset += mtk::oztcecgemm::detail::get_data_size_in_byte(data_type_list[1]) * m * n;
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}

	return offset;
}

template <class T>
void split_AB(
		mtk::oztcecgemm::handle_t handle,
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

	const auto b_offset = split_core(
			handle->working_memory_ptr,
			op_A,
			m, k,
			a_ptr, lda,
			split_config.matrix_a_split_types,
			mtk::oztcecgemm::detail::matrix_A,
			&two_to_alpha,
			handle->cuda_stream
			);

	split_core(
			reinterpret_cast<std::uint8_t*>(handle->working_memory_ptr) + b_offset,
			op_B,
			k, n,
			b_ptr, ldb,
			split_config.matrix_b_split_types,
			mtk::oztcecgemm::detail::matrix_B,
			&two_to_alpha,
			handle->cuda_stream
			);
}

cudaDataType_t to_cudaDataType_t(
		const mtk::oztcecgemm::detail::data_t d
		) {
	switch (d) {
	case mtk::oztcecgemm::detail::fp32:
		return CUDA_R_32F;
	case mtk::oztcecgemm::detail::fp16:
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

void gemm_core(
		mtk::oztcecgemm::handle_t handle,
		const mtk::oztcecgemm::operation_t op_A,
		const mtk::oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* const a_ptr, const std::size_t lda, const mtk::oztcecgemm::detail::data_t type_a,
		const void* const b_ptr, const std::size_t ldb, const mtk::oztcecgemm::detail::data_t type_b,
		const mtk::oztcecgemm::detail::gemm_pair_config_t& gemm_pair_config,
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	const auto gemm_mode = gemm_pair_config.gemm_mode;
	const auto split_config = mtk::oztcecgemm::detail::get_split_config(compute_mode);
	const auto lda_r = gemm_pair_config.A_id == -1 ? lda : k;
	const auto ldb_r = gemm_pair_config.B_id == -1 ? ldb : k;
	const void* const a_ptr_r = gemm_pair_config.A_id == -1 ? a_ptr : nullptr;
	const void* const b_ptr_r = gemm_pair_config.B_id == -1 ? a_ptr : nullptr;
	void* const c_ptr_r = nullptr;

	const float alpha_r = 1, beta_r = 0;

	switch (gemm_mode) {
	case mtk::oztcecgemm::detail::cublas_sgemm:
	case mtk::oztcecgemm::detail::cublas_bf16:
	case mtk::oztcecgemm::detail::cublas_tf32:
	case mtk::oztcecgemm::detail::cublas_fp16:
		{
			const auto op_A_r = gemm_pair_config.A_id == -1 ? to_cublasOperation_t(op_A) : CUBLAS_OP_T;
			const auto op_B_r = gemm_pair_config.B_id == -1 ? to_cublasOperation_t(op_B) : CUBLAS_OP_N;

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
						a_ptr_r, to_cudaDataType_t(type_a), lda_r,
						b_ptr_r, to_cudaDataType_t(type_b), ldb_r,
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
			const auto op_A_r = gemm_pair_config.A_id == -1 ? to_shgemm_operation_t(op_A) : mtk::shgemm::op_t;
			const auto op_B_r = gemm_pair_config.B_id == -1 ? to_shgemm_operation_t(op_B) : mtk::shgemm::op_n;
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
	case mtk::oztcecgemm::detail::fp16tcec:
	case mtk::oztcecgemm::detail::tf32tcec:
		{}
		break;
	default:
		OZTCECGEM_NOT_IMPLEMENTED;
	}
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
	mtk::oztcecgemm::detail::data_t input_type;
	if (compute_mode == mtk::oztcecgemm::fp32_split_3) {
		input_type = mtk::oztcecgemm::detail::fp32;
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}

	if (input_type == mtk::oztcecgemm::detail::fp32) {
		split_AB(
				handle,
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
					gemm_pair_config,
					compute_mode
					);
		}
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}

	return 0;
}
