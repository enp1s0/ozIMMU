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
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}

	return 0;
}
