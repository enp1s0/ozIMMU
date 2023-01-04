#include <cutf/thread.hpp>
#include <cutf/experimental/fp.hpp>
#include <cutf/math.hpp>
#include "config.hpp"
#include "split.hpp"

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
	T* local_ptr = ptr + threadIdx.x * inc;

	T local_abs_max = 0;

	unsigned i = 0;
	for (; i < length; i += blockDim.x) {
		const auto v = cutf::experimental::fp::reinterpret_as_fp(
				cutf::experimental::fp::reinterpret_as_uint(*local_ptr) &
				((~static_cast<bs_t>(0)) << cutf::experimental::fp::get_exponent_size<T>())
				);

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
template <bool col_major, class INPUT_T, class OUTPUT_1_T, class OUTPUT_2_T>
__global__ void split_2_no_smem_kernel(
		OUTPUT_1_T* const out_1_ptr,
		OUTPUT_2_T* const out_2_ptr,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		// alpha = ceil((24 + log2(n)) / 2)
		const INPUT_T two_to_alpha
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
		const auto a = in_ptr[(col_major ? row_index : (row_index * ld)) + (col_major ? i * ld : i)];
		const auto a1 = (sigma + a) - sigma;
		const auto a2 = a - a1;
		out_1_ptr[row_index * n + i] = a1;
		out_2_ptr[row_index * n + i] = a2;
	}
}

template <class INPUT_T, class OUTPUT_1_T, class OUTPUT_2_T>
void split_2_A(
		OUTPUT_1_T* const out_1_ptr,
		OUTPUT_2_T* const out_2_ptr,
		const std::size_t m,
		const std::size_t n,
		const INPUT_T* const in_ptr,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		// alpha = ceil((24 + log2(n)) / 2)
		const INPUT_T two_to_alpha,
		const cudaStream_t cuda_stream
		) {
	const std::size_t block_size = 256;
	const std::size_t grid_size = m;

	decltype(split_2_no_smem_kernel<true, INPUT_T, OUTPUT_1_T, OUTPUT_2_T>) kernel_func;

	if (op == mtk::oztcecgemm::op_n) {
		kernel_func = split_2_no_smem_kernel<true , INPUT_T, OUTPUT_1_T, OUTPUT_2_T>;
	} else {
		kernel_func = split_2_no_smem_kernel<false, INPUT_T, OUTPUT_1_T, OUTPUT_2_T>;
	}

	kernel_func<<<grid_size, block_size, 0, cuda_stream>>>(
			out_1_ptr,
			out_2_ptr,
			m, n,
			in_ptr,
			ld,
			two_to_alpha
			);
}
} // unnamed namespace

template <class INPUT_T, class OUTPUT_1_T, class OUTPUT_2_T>
void mtk::oztcecgemm::split_2(
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
		) {
	if (matrix == mtk::oztcecgemm::detail::matrix_A) {
		split_2_A(
				out_1_ptr,
				out_2_ptr,
				m, n,
				in_ptr,
				ld,
				op,
				two_to_alpha,
				cuda_stream
				);
	} else {
		split_2_A(
				out_1_ptr,
				out_2_ptr,
				n, m,
				in_ptr,
				ld,
				op == mtk::oztcecgemm::op_n ? mtk::oztcecgemm::op_t : mtk::oztcecgemm::op_n,
				two_to_alpha,
				cuda_stream
				);
	}
}
