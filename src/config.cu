#include "config.hpp"

mtk::oztcecgemm::detail::split_config_t mtk::oztcecgemm::detail::get_split_config(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::oztcecgemm::fp32_split_3:
		return split_config_t{{detail::fp16, detail::fp32}, {detail::fp16, detail::fp32}};
	default:
		break;
	}
	return split_config_t{{}, {}};
}

std::size_t mtk::oztcecgemm::detail::get_data_size_in_byte(
		const mtk::oztcecgemm::detail::data_t d
		) {
	switch (d) {
	case mtk::oztcecgemm::detail::fp32:
		return 4;
	case mtk::oztcecgemm::detail::fp16:
		return 2;
	default:
		break;
	}
	return 0;
}
