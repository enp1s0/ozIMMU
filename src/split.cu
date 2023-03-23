#include <cutf/thread.hpp>
#include <cutf/experimental/fp.hpp>
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/cuda.hpp>
#include "config.hpp"
#include "split.hpp"
#include "utils.hpp"

namespace {
template <class T>
__device__ T get_exp_max_element(
		// [length * inc]
		const T* const ptr,
		const unsigned length,
		const unsigned inc,
		// [blockDim.x / warp_size]
		typename mtk::ozimma::detail::real_type<T>::type* const working_smem_ptr
		) {
	using bs_t = typename cutf::experimental::fp::same_size_uint<T>::type;

	T local_abs_max = 0;

	unsigned i = threadIdx.x;
	const T* local_ptr = ptr + i * inc;
	for (; i < length; i += blockDim.x) {
		const auto v = cutf::experimental::fp::reinterpret_as_fp(cutf::experimental::fp::mask_exponent(*local_ptr));

		local_abs_max = cutf::math::max(local_abs_max, v);

		local_ptr += inc * blockDim.x;
	}

	// Inner-warp reduction
	for (std::uint32_t offset = cutf::thread::warp_size_const >> 1; offset >= 1; offset >>= 1) {
		local_abs_max = cutf::math::max(
				__shfl_xor_sync(~0u, local_abs_max, offset),
				local_abs_max
				);
	}

	// Inner-threadblock reduction
	if ((threadIdx.x & 0x1f) == 0) {
		working_smem_ptr[threadIdx.x >> 5] = local_abs_max;
	}

	__syncthreads();
	local_abs_max = 0;
	if (threadIdx.x < cutf::thread::warp_size_const) {
		if (threadIdx.x < (blockDim.x / cutf::thread::warp_size_const)) {
			local_abs_max = working_smem_ptr[threadIdx.x];
		}

		for (std::uint32_t offset = cutf::thread::warp_size_const >> 1; offset >= 1; offset >>= 1) {
			local_abs_max = cutf::math::max(
					__shfl_xor_sync(~0u, local_abs_max, offset),
					local_abs_max
					);
		}

		if (threadIdx.x == 0) {
			working_smem_ptr[0] = local_abs_max;
		}
	}

	__syncthreads();

	return working_smem_ptr[0];
}

template <>
__device__ cuDoubleComplex get_exp_max_element<cuDoubleComplex>(
		// [length * inc]
		const cuDoubleComplex* const ptr,
		const unsigned length,
		const unsigned inc,
		// [blockDim.x / warp_size]
		double* const working_smem_ptr
		) {
	using real_t = typename mtk::ozimma::detail::real_type<cuDoubleComplex>::type;
	using bs_t = typename cutf::experimental::fp::same_size_uint<real_t>::type;

	real_t local_real_abs_max = 0;
	real_t local_imag_abs_max = 0;

	unsigned i = threadIdx.x;
	const cuDoubleComplex* local_ptr = ptr + i * inc;
	for (; i < length; i += blockDim.x) {
		const auto v = *local_ptr;
		const auto real_exp = cutf::experimental::fp::reinterpret_as_fp(cutf::experimental::fp::mask_exponent(v.x));
		const auto imag_exp = cutf::experimental::fp::reinterpret_as_fp(cutf::experimental::fp::mask_exponent(v.y));

		local_real_abs_max = cutf::math::max(local_real_abs_max, real_exp);
		local_imag_abs_max = cutf::math::max(local_imag_abs_max, imag_exp);

		local_ptr += inc * blockDim.x;
	}

	// Inner-warp reduction
	for (std::uint32_t offset = cutf::thread::warp_size_const >> 1; offset >= 1; offset >>= 1) {
		local_real_abs_max = cutf::math::max(
				__shfl_xor_sync(~0u, local_real_abs_max, offset),
				local_real_abs_max
				);
		local_imag_abs_max = cutf::math::max(
				__shfl_xor_sync(~0u, local_imag_abs_max, offset),
				local_imag_abs_max
				);
	}

	// Inner-threadblock reduction
	if ((threadIdx.x & 0x1f) == 0) {
		working_smem_ptr[threadIdx.x >> 5] = local_real_abs_max;
	}

	__syncthreads();
	local_real_abs_max = 0;
	if (threadIdx.x < cutf::thread::warp_size_const) {
		if (threadIdx.x < (blockDim.x / cutf::thread::warp_size_const)) {
			local_real_abs_max = working_smem_ptr[threadIdx.x];
		}

		for (std::uint32_t offset = cutf::thread::warp_size_const >> 1; offset >= 1; offset >>= 1) {
			local_real_abs_max = cutf::math::max(
					__shfl_xor_sync(~0u, local_real_abs_max, offset),
					local_real_abs_max
					);
		}
	}
	__syncthreads();

	if ((threadIdx.x & 0x1f) == 0) {
		working_smem_ptr[threadIdx.x >> 5] = local_imag_abs_max;
	}

	__syncthreads();
	local_imag_abs_max = 0;
	if (threadIdx.x < cutf::thread::warp_size_const) {
		if (threadIdx.x < (blockDim.x / cutf::thread::warp_size_const)) {
			local_imag_abs_max = working_smem_ptr[threadIdx.x];
		}

		for (std::uint32_t offset = cutf::thread::warp_size_const >> 1; offset >= 1; offset >>= 1) {
			local_imag_abs_max = cutf::math::max(
					__shfl_xor_sync(~0u, local_imag_abs_max, offset),
					local_imag_abs_max
					);
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		working_smem_ptr[0] = local_real_abs_max;
		working_smem_ptr[1] = local_imag_abs_max;
	}

	__syncthreads();

	return make_cuDoubleComplex(working_smem_ptr[0], working_smem_ptr[1]);
}

template <class INPUT_T, class MANTISSA_T>
__device__ void cut_int8_core(
		std::int8_t* const out_ptr,
		const std::size_t inc,
		const INPUT_T a,
		const INPUT_T max_exp,
		const unsigned num_split,
		const unsigned mantissa_length
		) {
	const std::uint8_t sign_flag = a > 0;
	const auto mantissa = static_cast<MANTISSA_T>(cutf::experimental::fp::mask_mantissa(a) | (1lu << cutf::experimental::fp::get_mantissa_size<INPUT_T>()))
		<< ((sizeof(MANTISSA_T) - sizeof(INPUT_T)) * 8 + cutf::experimental::fp::get_exponent_size<INPUT_T>());
	const auto mantissa_shift_offset = (cutf::experimental::fp::reinterpret_as_uint(max_exp) - cutf::experimental::fp::mask_exponent(a)) >> cutf::experimental::fp::get_mantissa_size<INPUT_T>();

	auto shifted_mantissa = mantissa >> mantissa_shift_offset;
	for (unsigned s = 0; s < num_split; s++) {
		const std::int8_t int8 = static_cast<std::int8_t>(shifted_mantissa >> (sizeof(MANTISSA_T) * 8 - mantissa_length)) * (sign_flag ? 1 : -1);
		shifted_mantissa <<= mantissa_length;

		out_ptr[s * inc] = int8;
	}
}

__device__ cuDoubleComplex x2(const cuDoubleComplex a) {
	return make_cuDoubleComplex(a.x * 2, a.y * 2);
}

__device__ double x2(const double a) {
	return a * 2;
}

template <class INPUT_T, class MANTISSA_T>
__global__ void split_int8_kernel(
		std::int8_t* const out_ptr,
		typename mtk::ozimma::detail::real_type<INPUT_T>::type* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		const unsigned num_split,
		const unsigned mantissa_length,
		const bool col_major
		) {
	__shared__ typename mtk::ozimma::detail::real_type<INPUT_T>::type smem[32];
	const auto row_index = blockIdx.x;
	const auto max_exp = x2(get_exp_max_element(
			in_ptr + (col_major ? row_index : (row_index * ld)),
			n,
			(col_major ? ld : 1),
			smem
			));

	const auto N = m * n;
	for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
		const auto a = in_ptr[(col_major ? (i * ld + row_index) : (i + row_index * ld))];
		if constexpr (std::is_same<cuDoubleComplex, INPUT_T>::value) {
			cut_int8_core<double, MANTISSA_T>(out_ptr + row_index * n + i                , N, a.x, max_exp.x, num_split, mantissa_length);
			cut_int8_core<double, MANTISSA_T>(out_ptr + row_index * n + i + N * num_split, N, a.y, max_exp.y, num_split, mantissa_length);
		} else {
			cut_int8_core<double, MANTISSA_T>(out_ptr + row_index * n + i                , N, a  , max_exp  , num_split, mantissa_length);
		}
	}

	if (threadIdx.x == 0) {
		if constexpr (std::is_same<cuDoubleComplex, INPUT_T>::value) {
			max_exp_ptr[blockIdx.x    ] = max_exp.x;
			max_exp_ptr[blockIdx.x + m] = max_exp.y;
		} else {
			max_exp_ptr[blockIdx.x] = max_exp;
		}
	}
}

template <class INPUT_T>
void split_int8_A(
		std::int8_t* const out_ptr,
		typename mtk::ozimma::detail::real_type<INPUT_T>::type* const max_exp_ptr,
		const mtk::ozimma::operation_t op,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		const unsigned num_split,
		const unsigned mantissa_length,
		cudaStream_t cuda_stream
		) {
	const dim3 block_size = 256;
	const dim3 grid_size = m;

	const bool is_col_major = op == mtk::ozimma::op_n;

	using MANTISSA_T = __uint128_t;
	split_int8_kernel<INPUT_T, MANTISSA_T>
		<<<grid_size, block_size, 0, cuda_stream>>>(
				out_ptr,
				max_exp_ptr,
				m, n,
				in_ptr,
				ld,
				num_split,
				mantissa_length,
				is_col_major
				);
}

} // unnamed namespace

template <class T>
void mtk::ozimma::split_int8(
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
		) {
	if (matrix == mtk::ozimma::detail::matrix_A) {
		split_int8_A(
				out_ptr,
				max_exp_ptr,
				op,
				m, n,
				in_ptr, ld,
				num_split,
				bits_per_int8,
				cuda_stream
				);
	} else {
		split_int8_A(
				out_ptr,
				max_exp_ptr,
				op == mtk::ozimma::op_n ? mtk::ozimma::op_t : mtk::ozimma::op_n,
				n, m,
				in_ptr, ld,
				num_split,
				bits_per_int8,
				cuda_stream
				);
	}
}
// Instance
template
void mtk::ozimma::split_int8<double>(
		std::int8_t* const out_ptr,
		double* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const double* const in_ptr,
		const std::size_t ld,
		const mtk::ozimma::operation_t op,
		const mtk::ozimma::detail::matrix_t matrix,
		const unsigned num_split,
		const unsigned bits_per_int8,
		const cudaStream_t cuda_stream
		);
template
void mtk::ozimma::split_int8<cuDoubleComplex>(
		std::int8_t* const out_ptr,
		double* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const cuDoubleComplex* const in_ptr,
		const std::size_t ld,
		const mtk::ozimma::operation_t op,
		const mtk::ozimma::detail::matrix_t matrix,
		const unsigned num_split,
		const unsigned bits_per_int8,
		const cudaStream_t cuda_stream
		);
