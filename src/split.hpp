#pragma once
#include <cmath>
#include <unordered_map>
#include <cutf/experimental/fp.hpp>
#include <ozimma/ozimma.hpp>
#include "config.hpp"
#include "utils.hpp"
#include "handle.hpp"

namespace mtk {
namespace ozimma {
template <class T>
void split_int8(
		std::int8_t* const out_ptr,
		typename mtk::ozimma::detail::real_type<T>::type* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const T* const in_ptr,
		const std::size_t ld,
		const mtk::ozimma::operation_t op,
		const mtk::ozimma::detail::matrix_t matrix,
		const unsigned num_split,
		const unsigned bits_per_int8,
		const cudaStream_t cuda_stream
		);

template <class T>
std::unordered_map<mtk::ozimma::compute_mode_t, std::uint64_t> get_mantissa_loss_total(
		mtk::ozimma::handle& handle,
		const std::size_t m,
		const std::size_t n,
		const T* const in_ptr,
		const std::size_t ld,
		const mtk::ozimma::operation_t op,
		const unsigned bits_per_int8,
		const cudaStream_t cuda_stream,
		const bool download
		);

void init_mantissa_loss_counter(
		mtk::ozimma::handle& handle
		);

template <class INPUT_T>
INPUT_T get_two_to_alpha(const std::size_t k) {
	return 1lu << static_cast<unsigned>(std::ceil((cutf::experimental::fp::get_mantissa_size<INPUT_T>() + 1 + std::log2(k)) / 2));
}
} // namespace ozimma
} // namespace mtk
