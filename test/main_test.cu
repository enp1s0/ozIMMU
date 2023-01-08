#include <iostream>
#include <chrono>
#include <oztcecgemm/oztcecgemm.hpp>
#include <cutf/memory.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>

constexpr unsigned test_count = 100;

inline mtk::mateval::layout_t conv_layout_oz2mateval(const mtk::oztcecgemm::operation_t op) {
	if (op == mtk::oztcecgemm::op_n) {
		return mtk::mateval::col_major;
	}
	return mtk::mateval::row_major;
}

template <class T, class MATMUL_FUNC>
void gemm_eval_core(
		const mtk::oztcecgemm::operation_t op_a,
		const mtk::oztcecgemm::operation_t op_b,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const T* const a_ptr, const std::size_t lda,
		const T* const b_ptr, const std::size_t ldb,
		T* const c_ptr, const std::size_t ldc,
		const MATMUL_FUNC matmul_func,
		const std::string mode_name
		) {
	matmul_func(
			op_a, op_b,
			m, n, k,
			a_ptr, lda,
			b_ptr, ldb,
			c_ptr, ldc
			);

	const auto error = mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::relative_residual | mtk::mateval::max_relative_error,
			m, n, k,
			conv_layout_oz2mateval(op_a),
			conv_layout_oz2mateval(op_b),
			mtk::mateval::col_major,
			a_ptr, lda,
			b_ptr, ldb,
			c_ptr, ldc
			);

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto start_clock = std::chrono::system_clock::now();

	for (unsigned i = 0; i < test_count; i++) {
		matmul_func(
				op_a, op_b,
				m, n, k,
				a_ptr, lda,
				b_ptr, ldb,
				c_ptr, ldc
				);
	}

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / test_count;

	const auto throughput = 2 * m * n * k / elapsed_time;

	std::printf("%s,%lu,%lu,%lu,%e,%e,%e\n",
			mode_name.c_str(),
			m, n, k,
			error.at(mtk::mateval::relative_residual),
			error.at(mtk::mateval::max_relative_error),
			throughput * 1e-12
			);
	std::fflush(stdout);
}

template <class T>
void gemm_eval(
		const mtk::oztcecgemm::gemm_list_t& gemm_list
		) {
	mtk::oztcecgemm::handle_t oztcecgemm_handle;
	mtk::oztcecgemm::create(&oztcecgemm_handle);
	mtk::oztcecgemm::reallocate_working_memory(oztcecgemm_handle, gemm_list);

	std::size_t max_AB_size = 0;
	std::size_t max_C_size = 0;
	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		max_AB_size = std::max(max_AB_size, m * k + k * n);
		max_C_size  = std::max(max_C_size , m * n);
	}

	auto mat_AB_uptr = cutf::memory::get_device_unique_ptr<T>(max_AB_size);
	auto mat_C_uptr  = cutf::memory::get_device_unique_ptr<T>(max_C_size);

	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), 0));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_AB_uptr.get(), max_AB_size));

	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		const auto mode = std::get<3>(gemm);
		gemm_eval_core(
				mtk::oztcecgemm::op_n,
				mtk::oztcecgemm::op_n,
				m, n, k,
				mat_AB_uptr.get(), m,
				mat_AB_uptr.get() + m * k, k,
				mat_C_uptr.get(), m,
				[&](
						const mtk::oztcecgemm::operation_t op_a,
						const mtk::oztcecgemm::operation_t op_b,
						const std::size_t m,
						const std::size_t n,
						const std::size_t k,
						const T* const a_ptr, const std::size_t lda,
						const T* const b_ptr, const std::size_t ldb,
						T* const c_ptr, const std::size_t ldc
									) {
					const T alpha = 1, beta = 0;
					mtk::oztcecgemm::gemm(
							oztcecgemm_handle,
							op_a, op_b,
							m, n, k,
							&alpha,
							a_ptr, lda,
							b_ptr, ldb,
							&beta,
							c_ptr, ldc,
							mode
							);
				},
				mtk::oztcecgemm::get_compute_mode_name_str(mode)
				);
	}

	mtk::oztcecgemm::destroy(oztcecgemm_handle);
}

int main(int argc, char** argv) {
	mtk::oztcecgemm::gemm_list_t gemm_list;

	gemm_list.push_back(std::make_tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>(
				1024,
				1024,
				1024,
				mtk::oztcecgemm::fp32_split_3
				));

	gemm_eval<float>(gemm_list);
}
