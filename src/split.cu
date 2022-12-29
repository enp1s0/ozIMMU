#include <cutf/thread.hpp>
#include <cutf/experimental/fp.hpp>
#include <cutf/math.hpp>
#include "config.hpp"

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
}
