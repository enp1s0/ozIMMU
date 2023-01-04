#pragma once
#include <cmath>
#include <cutf/experimental/fp.hpp>
#include "config.hpp"

namespace mtk {
namespace oztcecgemm {
template <class INPUT_T, class OUTPUT_1_T, class OUTPUT_2_T>
void split_2(
		OUTPUT_1_T* const out_1_ptr,
		OUTPUT_2_T* const out_2_ptr,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		// alpha = ceil((24 + log2(n)) / 2)
		const INPUT_T two_to_alpha,
		const cudaStream_t cuda_stream
		);

template <class INPUT_T>
INPUT_T get_two_to_alpha(const std::size_t k) {
	return std::ceil((cutf::experimental::fp::get_mantissa_size<INPUT_T>() + 1 + std::log2(k)) / 2);
}
} // namespace oztcecgemm
} // namespace mtk
