#include "config.hpp"
#include "utils.hpp"

mtk::oztcecgemm::detail::split_config_t mtk::oztcecgemm::detail::get_split_config(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::oztcecgemm::sgemm:
		return split_config_t {
			{detail::original},
			{detail::original},
			{{0, 0, detail::cublas_sgemm}}
		};
	case mtk::oztcecgemm::fp32_split_3:
		return split_config_t {
			{detail::original, detail::fp16, detail::fp32},
			{detail::original, detail::fp16, detail::fp32},
			{{1, 1, detail::cublas_tf32}, {1, 2, detail::shgemm_tf32}, {2, 0, detail::cublas_sgemm}}
		};
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
