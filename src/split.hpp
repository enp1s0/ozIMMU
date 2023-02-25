#pragma once
#include <cmath>
#include <cutf/experimental/fp.hpp>
#include "config.hpp"

namespace mtk {
namespace oztcecgemm {
template <class T>
void split_int8(
		std::int8_t* const out_ptr,
		T* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const T* const in_ptr,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		const unsigned num_split,
		const unsigned bits_per_int8,
		const cudaStream_t cuda_stream
		);

template <class INPUT_T>
INPUT_T get_two_to_alpha(const std::size_t k) {
	return 1lu << static_cast<unsigned>(std::ceil((cutf::experimental::fp::get_mantissa_size<INPUT_T>() + 1 + std::log2(k)) / 2));
}
} // namespace oztcecgemm
} // namespace mtk
