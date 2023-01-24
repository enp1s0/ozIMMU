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

std::string mtk::oztcecgemm::detail::gemm_mode_str(
		const mtk::oztcecgemm::detail::gemm_t gemm_mode
		) {
#define GEMM_MODE_STR_CASE(mode) case mode: return #mode
	switch (gemm_mode) {
	GEMM_MODE_STR_CASE(cublas_sgemm);
	GEMM_MODE_STR_CASE(cublas_tf32 );
	GEMM_MODE_STR_CASE(cublas_fp16 );
	GEMM_MODE_STR_CASE(cublas_bf16 );
	GEMM_MODE_STR_CASE(tf32tcec    );
	GEMM_MODE_STR_CASE(fp16tcec    );
	GEMM_MODE_STR_CASE(shgemm_fp16 );
	GEMM_MODE_STR_CASE(shgemm_tf32 );
	GEMM_MODE_STR_CASE(hsgemm_fp16 );
	GEMM_MODE_STR_CASE(hsgemm_tf32 );
	default:
		break;
	}
	return "Unknown";
}
