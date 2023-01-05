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
	fp32,
	fp16,
	int8
};

std::size_t get_data_size_in_byte(
		const data_t d
		);

template <class T>
inline data_t get_data_t();
template <>
inline data_t get_data_t<float>() {return data_t::fp32;}

struct split_config_t {
	std::vector<data_t> matrix_a_split_types;
	std::vector<data_t> matrix_b_split_types;
};

split_config_t get_split_config(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		);

} // namespace detail
} // namespace oztcecgemm
} // namespace mtk
