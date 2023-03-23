#include "config.hpp"
#include "utils.hpp"

mtk::ozimma::detail::split_config_t mtk::ozimma::detail::get_split_config(
		const mtk::ozimma::compute_mode_t compute_mode
		) {
	switch (compute_mode) {
	case mtk::ozimma::sgemm:
		return split_config_t {
			{original},
			{original},
			{{0, 0, detail::cublas_sgemm}}
		};
	case mtk::ozimma::dgemm:
		return split_config_t {
			{original},
			{original},
			{{0, 0, detail::cublas_dgemm}}
		};
	case mtk::ozimma::fp64_int8_6:
	case mtk::ozimma::fp64_int8_7:
	case mtk::ozimma::fp64_int8_8:
	case mtk::ozimma::fp64_int8_9:
	case mtk::ozimma::fp64_int8_10:
	case mtk::ozimma::fp64_int8_11:
	case mtk::ozimma::fp64_int8_12:
	case mtk::ozimma::fp64_int8_13:
		{
			unsigned num_split = 0;
			if (compute_mode == mtk::ozimma::fp64_int8_6 ) {num_split = 6;}
			if (compute_mode == mtk::ozimma::fp64_int8_7 ) {num_split = 7;}
			if (compute_mode == mtk::ozimma::fp64_int8_8 ) {num_split = 8;}
			if (compute_mode == mtk::ozimma::fp64_int8_9 ) {num_split = 9;}
			if (compute_mode == mtk::ozimma::fp64_int8_10) {num_split = 10;}
			if (compute_mode == mtk::ozimma::fp64_int8_11) {num_split = 11;}
			if (compute_mode == mtk::ozimma::fp64_int8_12) {num_split = 12;}
			if (compute_mode == mtk::ozimma::fp64_int8_13) {num_split = 13;}

			// Data
			std::vector<mtk::ozimma::data_t> split_types(num_split + 1);
			std::fill(split_types.begin(), split_types.end(), mtk::ozimma::int8);
			split_types[0] = mtk::ozimma::original;

			// Computation
			std::vector<mtk::ozimma::detail::gemm_pair_config_t> gemm_pair_list;
			for (int sum = 2; sum <= num_split + 1; sum++) {
				for (int j = 1; j < sum; j++) {
					if (j > num_split || sum - j > num_split)
						continue;
					gemm_pair_list.push_back({j, sum - j, mtk::ozimma::detail::int8tc});
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

std::string mtk::ozimma::detail::gemm_mode_str(
		const mtk::ozimma::detail::gemm_t gemm_mode
		) {
#define GEMM_MODE_STR_CASE(mode) case mode: return #mode
	switch (gemm_mode) {
	GEMM_MODE_STR_CASE(cublas_sgemm);
	GEMM_MODE_STR_CASE(cublas_dgemm);
	GEMM_MODE_STR_CASE(cublas_tf32 );
	GEMM_MODE_STR_CASE(cublas_fp16 );
	GEMM_MODE_STR_CASE(cublas_bf16 );
	GEMM_MODE_STR_CASE(int8tc      );
	default:
		break;
	}
	return "Unknown";
}

// working memory size calculation
std::size_t mtk::ozimma::detail::calculate_working_memory_size(
		const std::size_t m,
		const std::size_t n,
		const mtk::ozimma::compute_mode_t compute_mode,
		const mtk::ozimma::detail::matrix_t matrix,
		const mtk::ozimma::element_kind_t element_kind
		) {
	const auto split_config = mtk::ozimma::detail::get_split_config(compute_mode);

	decltype(split_config.matrix_A_split_types) split_data_types;
	if (matrix == mtk::ozimma::detail::matrix_A) {
		split_data_types = split_config.matrix_A_split_types;
	} else {
		split_data_types = split_config.matrix_B_split_types;
	}

	std::size_t sum_data_type_size = 0;
	for (const auto d : split_data_types) {
		sum_data_type_size += mtk::ozimma::get_data_size_in_byte(d);
	}

	return sum_data_type_size * m * n * (element_kind == mtk::ozimma::real ? 1 : 2);
}
