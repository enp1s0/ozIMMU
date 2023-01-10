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
		void* const out_1_ptr, const mtk::oztcecgemm::detail::data_t type_1,
		void* const out_2_ptr, const mtk::oztcecgemm::detail::data_t type_2,
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

	if (type_1 == mtk::oztcecgemm::detail::fp16 && type_2 == mtk::oztcecgemm::detail::fp32) {
		CUTF_CHECK_ERROR(cudaLaunchKernel((void*)split_2_no_smem_kernel<INPUT_T, half, float>, grid_size, block_size, (void**)args, 0, cuda_stream));
	} else {
		OZTCECGEM_NOT_IMPLEMENTED;
	}
}
} // unnamed namespace

void mtk::oztcecgemm::split_2(
		void* const out_1_ptr, const mtk::oztcecgemm::detail::data_t type_1,
		void* const out_2_ptr, const mtk::oztcecgemm::detail::data_t type_2,
		const std::size_t m,
		const std::size_t n,
		const void* const in_ptr, const mtk::oztcecgemm::detail::data_t type_in,
		const std::size_t ld,
		const mtk::oztcecgemm::operation_t op,
		const mtk::oztcecgemm::detail::matrix_t matrix,
		// alpha = ceil((24 + log2(n)) / 2)
		const void* two_to_alpha,
		const cudaStream_t cuda_stream
		) {
	if (matrix == mtk::oztcecgemm::detail::matrix_A) {
		if (type_in == mtk::oztcecgemm::detail::fp32) {
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
		if (type_in == mtk::oztcecgemm::detail::fp32) {
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
