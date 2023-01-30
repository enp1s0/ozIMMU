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
		T* const working_smem_ptr
		) {
	using bs_t = typename cutf::experimental::fp::same_size_uint<T>::type;

	T local_abs_max = 0;

	unsigned i = threadIdx.x;
	const T* local_ptr = ptr + i * inc;
	for (; i < length; i += blockDim.x) {
		const auto v = cutf::experimental::fp::reinterpret_as_fp(cutf::experimental::fp::mask_exponent(*local_ptr));

		local_abs_max = cutf::math::max(local_abs_max, v);

		local_ptr += inc;
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

// This function splits floating-point values of a given input matrix.
// Matrix A is assumed.
template <class INPUT_T, class OUTPUT_1_T, class OUTPUT_2_T>
__global__ void split_2_no_smem_kernel(
		OUTPUT_1_T* const out_1_ptr,
		OUTPUT_2_T* const out_2_ptr,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		// alpha = ceil((24 + log2(n)) / 2)
		const INPUT_T two_to_alpha,
		const bool col_major
		) {
	__shared__ INPUT_T smem[32];
	const auto row_index = blockIdx.x;
	const auto max_exp = get_exp_max_element(
			in_ptr + (col_major ? row_index : (row_index * ld)),
			n,
			(col_major ? ld : 1),
			smem
			);
	const auto sigma = max_exp * 2 * 3 / 4 * two_to_alpha;

	for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
		const auto a = in_ptr[(col_major ? (i * ld + row_index) : (i + row_index * ld))];
		const OUTPUT_1_T a1 = (sigma + a) - sigma;
		const OUTPUT_2_T a2 = cutf::type::cast<INPUT_T>(a) - cutf::type::cast<INPUT_T>(a1);
		out_1_ptr[row_index * n + i] = a1;
		out_2_ptr[row_index * n + i] = a2;
	}
}

template <class INPUT_T>
void split_2_A(
		void* const out_1_ptr, const mtk::oztcecgemm::data_t type_1,
		void* const out_2_ptr, const mtk::oztcecgemm::data_t type_2,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		// alpha = ceil((24 + log2(n)) / 2)
		const INPUT_T two_to_alpha,
		const cudaStream_t cuda_stream
		) {
	const dim3 block_size = 256;
	const dim3 grid_size = m;

	const bool col_major = op == mtk::oztcecgemm::op_n;

	const void* args[] = {
		&out_1_ptr,
		&out_2_ptr,
		&m,
		&n,
		&in_ptr,
		&ld,
		&two_to_alpha,
		&col_major,
		nullptr
	};

	if (type_1 == mtk::oztcecgemm::fp16 && type_2 == mtk::oztcecgemm::fp32) {
		CUTF_CHECK_ERROR(cudaLaunchKernel((void*)split_2_no_smem_kernel<INPUT_T, half, float>, grid_size, block_size, (void**)args, 0, cuda_stream));
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}
}

template <class INPUT_T, class MANTISSA_T>
__global__ void split_int8_kernel(
		std::int8_t* const out_ptr,
		INPUT_T* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		const unsigned num_split,
		const unsigned mantissa_length,
		const bool col_major
		) {
	__shared__ INPUT_T smem[32];
	const auto row_index = blockIdx.x;
	const auto max_exp = get_exp_max_element(
			in_ptr + (col_major ? row_index : (row_index * ld)),
			n,
			(col_major ? ld : 1),
			smem
			);

	const auto N = m * n;
	for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
		const auto a = in_ptr[(col_major ? (i * ld + row_index) : (i + row_index * ld))];
		const std::uint8_t sign_flag = a > 0;
		const auto mantissa = static_cast<MANTISSA_T>(cutf::experimental::fp::mask_mantissa(a) | (1lu << cutf::experimental::fp::get_mantissa_size<INPUT_T>()))
			<< ((sizeof(MANTISSA_T) - sizeof(INPUT_T)) * 8 + cutf::experimental::fp::get_exponent_size<INPUT_T>());
		const auto mantissa_shift_offset = (cutf::experimental::fp::reinterpret_as_uint(max_exp) - cutf::experimental::fp::mask_exponent(a)) >> cutf::experimental::fp::get_mantissa_size<INPUT_T>();

		auto shifted_mantissa = mantissa >> mantissa_shift_offset;
		for (unsigned s = 0; s < num_split; s++) {
			const std::int8_t int8 = static_cast<std::int8_t>(shifted_mantissa >> (sizeof(MANTISSA_T) * 8 - mantissa_length)) * (sign_flag ? 1 : -1);
			shifted_mantissa <<= mantissa_length;

			out_ptr[row_index * n + i + s * N] = int8;
		}
	}

	if (threadIdx.x == 0) {
		max_exp_ptr[blockIdx.x] = max_exp;
	}
}

template <class INPUT_T>
void split_int8_A(
		std::int8_t* const out_ptr,
		INPUT_T* const max_exp_ptr,
		const mtk::oztcecgemm::operation_t op,
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

	const bool is_col_major = op == mtk::oztcecgemm::op_n;

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

void mtk::oztcecgemm::split_2(
		void* const out_1_ptr, const mtk::oztcecgemm::data_t type_1,
		void* const out_2_ptr, const mtk::oztcecgemm::data_t type_2,
		const std::size_t m,
		const std::size_t n,
		const void* const in_ptr, const mtk::oztcecgemm::data_t type_in,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		// alpha = ceil((24 + log2(n)) / 2)
		const void* two_to_alpha,
		const cudaStream_t cuda_stream
		) {
	if (matrix == mtk::oztcecgemm::detail::matrix_A) {
		if (type_in == mtk::oztcecgemm::fp32) {
			split_2_A(
					out_1_ptr, type_1,
					out_2_ptr, type_2,
					m, n,
					reinterpret_cast<const float*>(in_ptr),
					ld,
					op,
					*reinterpret_cast<const float*>(two_to_alpha),
					cuda_stream
					);
		} else {
			OZTCECGEM_NOT_IMPLEMENTED;
		}
	} else {
		if (type_in == mtk::oztcecgemm::fp32) {
			split_2_A(
					out_1_ptr, type_1,
					out_2_ptr, type_2,
					n, m,
					reinterpret_cast<const float*>(in_ptr),
					ld,
					op == mtk::oztcecgemm::op_n ? mtk::oztcecgemm::op_t : mtk::oztcecgemm::op_n,
					*reinterpret_cast<const float*>(two_to_alpha),
					cuda_stream
					);
		} else {
			OZTCECGEM_NOT_IMPLEMENTED;
		}
	}
}

template <class T>
void mtk::oztcecgemm::split_int8(
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
		) {
	if (matrix == mtk::oztcecgemm::detail::matrix_A) {
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
				op == mtk::oztcecgemm::op_n ? mtk::oztcecgemm::op_t : mtk::oztcecgemm::op_n,
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
void mtk::oztcecgemm::split_int8<double>(
		std::int8_t* const out_ptr,
		double* const max_exp_ptr,
		const std::size_t m,
		const std::size_t n,
		const double* const in_ptr,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		const unsigned num_split,
		const unsigned bits_per_int8,
		const cudaStream_t cuda_stream
		);
