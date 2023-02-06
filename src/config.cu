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
	case mtk::oztcecgemm::dgemm:
		return split_config_t {
			{original},
			{original},
			{{0, 0, detail::cublas_dgemm}}
		};
	case mtk::oztcecgemm::fp32_split_3:
		return split_config_t {
			{original, fp16, fp32},
			{original, fp16, fp32},
			{{1, 1, detail::cublas_fp16}, {1, 2, detail::hsgemm_fp16}, {2, 0, detail::fp16tcec}}
		};
	case mtk::oztcecgemm::fp64_int8_6:
	case mtk::oztcecgemm::fp64_int8_7:
	case mtk::oztcecgemm::fp64_int8_8:
	case mtk::oztcecgemm::fp64_int8_9:
		{
			unsigned num_split = 0;
			if (compute_mode == mtk::oztcecgemm::fp64_int8_6) {num_split = 6;}
			if (compute_mode == mtk::oztcecgemm::fp64_int8_7) {num_split = 7;}
			if (compute_mode == mtk::oztcecgemm::fp64_int8_8) {num_split = 8;}
			if (compute_mode == mtk::oztcecgemm::fp64_int8_9) {num_split = 9;}

			// Data
			std::vector<mtk::oztcecgemm::data_t> split_types(num_split + 1);
			std::fill(split_types.begin(), split_types.end(), mtk::oztcecgemm::int8);
			split_types[0] = mtk::oztcecgemm::original;

			// Computation
			std::vector<mtk::oztcecgemm::detail::gemm_pair_config_t> gemm_pair_list;
			for (int sum = 2; sum <= num_split + 1; sum++) {
				for (int j = 1; j < sum; j++) {
					gemm_pair_list.push_back({j, sum - j, mtk::oztcecgemm::detail::int8tc});
				}
			}

			return split_config_t {
				split_types,
					split_types,
					gemm_pair_list
			};
		}
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
	GEMM_MODE_STR_CASE(cublas_dgemm);
	GEMM_MODE_STR_CASE(cublas_tf32 );
	GEMM_MODE_STR_CASE(cublas_fp16 );
	GEMM_MODE_STR_CASE(cublas_bf16 );
	GEMM_MODE_STR_CASE(tf32tcec    );
	GEMM_MODE_STR_CASE(fp16tcec    );
	GEMM_MODE_STR_CASE(shgemm_fp16 );
	GEMM_MODE_STR_CASE(shgemm_tf32 );
	GEMM_MODE_STR_CASE(hsgemm_fp16 );
	GEMM_MODE_STR_CASE(hsgemm_tf32 );
	GEMM_MODE_STR_CASE(int8tc      );
	default:
		break;
	}
	return "Unknown";
}
