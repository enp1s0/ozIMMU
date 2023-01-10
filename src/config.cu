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
			{{1, 1, detail::cublas_fp16}, {1, 2, detail::hsgemm_tf32}, {2, 0, detail::cublas_sgemm}}
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
	case mtk::oztcecgemm::detail::fp64:
		return 8;
	case mtk::oztcecgemm::detail::fp32:
		return 4;
	case mtk::oztcecgemm::detail::fp16:
		return 2;
	case mtk::oztcecgemm::detail::original:
		return 0;
	default:
		break;
	}
	return 0;
}

mtk::oztcecgemm::detail::data_t mtk::oztcecgemm::detail::get_output_type(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::oztcecgemm::sgemm:
		return mtk::oztcecgemm::detail::fp32;
	case mtk::oztcecgemm::fp32_split_3:
		return mtk::oztcecgemm::detail::fp64;
	default:
		break;
	}
	OZTCECGEM_NOT_IMPLEMENTED;
	return mtk::oztcecgemm::detail::original;
}
