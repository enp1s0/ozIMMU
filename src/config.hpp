#pragma once
#include <vector>
#include <oztcecgemm/oztcecgemm.hpp>

namespace mtk {
namespace oztcecgemm {
namespace detail {

enum matrix_t {
	matrix_A,
	matrix_B,
	matrix_C
};

enum data_t {
	fp64,
	fp32,
	fp16,
	int8,
	original
};

enum gemm_t {
	cublas_sgemm,
	cublas_tf32,
	cublas_fp16,
	cublas_bf16,
	tf32tcec,
	fp16tcec,
	shgemm_fp16,
	shgemm_tf32
};

std::size_t get_data_size_in_byte(
		const data_t d
		);

template <class T>
inline data_t get_data_t();
template <>
inline data_t get_data_t<float>() {return data_t::fp32;}

struct gemm_pair_config_t {
	// '-1' means the original matrix
	int A_id, B_id;
	gemm_t gemm_mode;
};

struct split_config_t {
	// {[0] = original_type, [1] = ...}
	std::vector<data_t> matrix_A_split_types;
	std::vector<data_t> matrix_B_split_types;
	std::vector<gemm_pair_config_t> gemm_pair_config_list;
};

split_config_t get_split_config(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		);

} // namespace detail
} // namespace oztcecgemm
} // namespace mtk
