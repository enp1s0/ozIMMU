#include "config.hpp"
#include "utils.hpp"

mtk::oztcecgemm::detail::split_config_t mtk::oztcecgemm::detail::get_split_config(
		const mtk::oztcecgemm::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::oztcecgemm::sgemm:
		return split_config_t {
			{original},
			{original},
			{{0, 0, detail::cublas_sgemm}}
		};
	case mtk::oztcecgemm::fp32_split_3:
		return split_config_t {
			{original, fp16, fp32},
			{original, fp16, fp32},
			{{1, 1, detail::cublas_fp16}, {1, 2, detail::hsgemm_fp16}, {2, 0, detail::fp16tcec}}
		};
	default:
		break;
	}
	return split_config_t{{}, {}};
}
