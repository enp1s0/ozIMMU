#include <iostream>
#include <chrono>
#include <oztcecgemm/oztcecgemm.hpp>
#include <cutf/memory.hpp>
#include <cutf/curand.hpp>
#include <mateval/comparison_cuda.hpp>
#include <matfile/matfile.hpp>

constexpr unsigned test_count = 100;

constexpr unsigned long long seed = 0;

inline mtk::mateval::layout_t conv_layout_oz2mateval(const mtk::oztcecgemm::operation_t op) {
	if (op == mtk::oztcecgemm::op_n) {
		return mtk::mateval::col_major;
	}
	return mtk::mateval::row_major;
}

template <class T>
__global__ void adjust_urand_kernel(
		T* const ptr,
		const T min_urand,
		const T max_urand,
		const std::size_t n
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}

	const auto v = ptr[tid];
	ptr[tid] = v * (max_urand - min_urand) + min_urand;
}

template <class T>
void adjust_urand(
		T* const ptr,
		const T min_urand,
		const T max_urand,
		const std::size_t n
		) {
	const auto block_size = 256lu;
	const auto grid_size = (n + block_size - 1) / block_size;

	adjust_urand_kernel<T><<<grid_size, block_size>>>(
			ptr,
			min_urand, max_urand,
			n
			);
}

template <class C_T, class AB_T, class MATMUL_FUNC>
void gemm_eval_core(
		const mtk::oztcecgemm::operation_t op_a,
		const mtk::oztcecgemm::operation_t op_b,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const AB_T* const a_ptr, const std::size_t lda,
		const AB_T* const b_ptr, const std::size_t ldb,
		C_T* const c_ptr, const std::size_t ldc,
		const MATMUL_FUNC matmul_func,
		const mtk::oztcecgemm::compute_mode_t mode
		) {
	matmul_func(
			op_a, op_b,
			m, n, k,
			a_ptr, lda,
			b_ptr, ldb,
			c_ptr, ldc
			);

	mtk::mateval::error_map_t error;
	if (mtk::oztcecgemm::get_output_type(mode) == mtk::oztcecgemm::fp32) {
		error = mtk::mateval::cuda::get_error_AxB(
				mtk::mateval::relative_residual | mtk::mateval::max_relative_error,
				m, n, k,
				conv_layout_oz2mateval(op_a),
				conv_layout_oz2mateval(op_b),
				mtk::mateval::col_major,
				a_ptr, lda,
				b_ptr, ldb,
				reinterpret_cast<float*>(c_ptr), ldc
				);
	} else {
		error = mtk::mateval::cuda::get_error_AxB(
				mtk::mateval::relative_residual | mtk::mateval::max_relative_error,
				m, n, k,
				conv_layout_oz2mateval(op_a),
				conv_layout_oz2mateval(op_b),
				mtk::mateval::col_major,
				a_ptr, lda,
				b_ptr, ldb,
				reinterpret_cast<double*>(c_ptr), ldc
				);
	}

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
			mtk::oztcecgemm::get_compute_mode_name_str(mode).c_str(),
			m, n, k,
			error.at(mtk::mateval::relative_residual),
			error.at(mtk::mateval::max_relative_error),
			throughput * 1e-12
			);
	std::fflush(stdout);
}

template <class T>
void gemm_eval(
		const mtk::oztcecgemm::gemm_list_t& gemm_list,
		const std::string input_mode
		) {
	mtk::oztcecgemm::handle_t oztcecgemm_handle;
	mtk::oztcecgemm::create(&oztcecgemm_handle);
	mtk::oztcecgemm::reallocate_working_memory(oztcecgemm_handle, gemm_list);

	std::size_t max_AB_count = 0;
	std::size_t max_C_size = 0;
	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		max_AB_count = std::max(max_AB_count, m * k + k * n);
		max_C_size  = std::max(max_C_size , m * n *
				mtk::oztcecgemm::get_data_size_in_byte(
				mtk::oztcecgemm::get_output_type(std::get<3>(gemm))));
	}

	auto mat_AB_uptr = cutf::memory::get_device_unique_ptr<T>(max_AB_count);
	auto mat_C_uptr  = cutf::memory::get_device_unique_ptr<std::uint8_t>(max_C_size);

	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	if (input_mode == "normal01") {
		CUTF_CHECK_ERROR(cutf::curand::generate_normal(*cugen.get(), mat_AB_uptr.get(), max_AB_count, 0, 1));
	} else {
		CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), mat_AB_uptr.get(), max_AB_count));
	}

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
						void* const c_ptr, const std::size_t ldc
									) {
					if (mtk::oztcecgemm::get_output_type(mode) == mtk::oztcecgemm::fp32) {
						using C_T = float;
						const C_T alpha = 1, beta = 0;
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
					} else {
						using C_T = double;
						const C_T alpha = 1, beta = 0;
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
					}
				},
				mode
				);
	}

	mtk::oztcecgemm::destroy(oztcecgemm_handle);
}

template <class SRC_T, class DST_T>
__global__ void vector_copy_kernel(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr,
		const std::size_t N
		) {
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= N) {
		return;
	}

	dst_ptr[tid] = src_ptr[tid];
}

template <class DEVICE_T>
void matfile_to_device_memory(
		DEVICE_T* const d_ptr,
		const std::string matfile_path
		) {
	std::size_t m, n;
	mtk::matfile::load_size(m, n, matfile_path);
	const auto dtype = mtk::matfile::load_dtype(matfile_path);

	auto h_mat_uptr = cutf::memory::get_host_unique_ptr<std::uint8_t>(m * n * mtk::matfile::get_dtype_size(dtype));

	mtk::matfile::load_dense(
			h_mat_uptr.get(),
			m,
			matfile_path
			);

	const std::size_t block_size = 256;
	const std::size_t grid_size = (m * n + block_size - 1) / block_size;

	if (dtype == mtk::matfile::fp32) {
		vector_copy_kernel<<<grid_size, block_size>>>(
				d_ptr,
				reinterpret_cast<float*>(h_mat_uptr.get()),
				m * n
				);
	} else if (dtype == mtk::matfile::fp64) {
		vector_copy_kernel<<<grid_size, block_size>>>(
				d_ptr,
				reinterpret_cast<double*>(h_mat_uptr.get()),
				m * n
				);
	} else {
		vector_copy_kernel<<<grid_size, block_size>>>(
				d_ptr,
				reinterpret_cast<long double*>(h_mat_uptr.get()),
				m * n
				);
	}
}

template <class T>
void gemm_eval_matfile(
		const mtk::oztcecgemm::gemm_list_t& gemm_list,
		const std::string matfile_A_path,
		const std::string matfile_B_path,
		const std::string matfile_C_path
		) {
	mtk::oztcecgemm::handle_t oztcecgemm_handle;
	mtk::oztcecgemm::create(&oztcecgemm_handle);
	mtk::oztcecgemm::reallocate_working_memory(oztcecgemm_handle, gemm_list);

	std::size_t max_AB_count = 0;
	std::size_t max_C_size = 0;
	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		max_AB_count = std::max(max_AB_count, m * k + k * n);
		max_C_size  = std::max(max_C_size , m * n *
				mtk::oztcecgemm::get_data_size_in_byte(
				mtk::oztcecgemm::get_output_type(std::get<3>(gemm))));
	}

	auto mat_AB_uptr = cutf::memory::get_device_unique_ptr<T>(max_AB_count);
	auto mat_C_uptr  = cutf::memory::get_device_unique_ptr<std::uint8_t>(max_C_size);
	auto mat_ref_uptr  = cutf::memory::get_host_unique_ptr<std::uint8_t>(max_C_size);


	for (const auto gemm : gemm_list) {
		const auto m = std::get<0>(gemm);
		const auto n = std::get<1>(gemm);
		const auto k = std::get<2>(gemm);
		const auto mode = std::get<3>(gemm);

		const auto a_ptr = mat_AB_uptr.get();
		const auto b_ptr = mat_AB_uptr.get() + m * k;
		const auto c_ptr = mat_C_uptr.get();

		matfile_to_device_memory(a_ptr, matfile_A_path);
		matfile_to_device_memory(b_ptr, matfile_B_path);
		mtk::matfile::load_dense(mat_ref_uptr.get(), m, matfile_C_path);

		mtk::mateval::error_map_t error;
		if (mtk::oztcecgemm::get_output_type(mode) == mtk::oztcecgemm::fp32) {
			using C_T = float;
			const C_T alpha = 1, beta = 0;
			mtk::oztcecgemm::gemm(
					oztcecgemm_handle,
					mtk::oztcecgemm::op_n, mtk::oztcecgemm::op_n,
					m, n, k,
					&alpha,
					a_ptr, m,
					b_ptr, k,
					&beta,
					c_ptr, m,
					mode
					);
			error = mtk::mateval::cuda::get_error(
					mtk::mateval::max_relative_error | mtk::mateval::relative_residual,
					m, n,
					mtk::mateval::col_major,
					mtk::mateval::col_major,
					c_ptr, m,
					reinterpret_cast<C_T*>(mat_ref_uptr.get()), m
					);
		} else {
			using C_T = double;
			const C_T alpha = 1, beta = 0;
			mtk::oztcecgemm::gemm(
					oztcecgemm_handle,
					mtk::oztcecgemm::op_n, mtk::oztcecgemm::op_n,
					m, n, k,
					&alpha,
					a_ptr, m,
					b_ptr, k,
					&beta,
					c_ptr, m,
					mode
					);
			error = mtk::mateval::cuda::get_error(
					mtk::mateval::max_relative_error | mtk::mateval::relative_residual,
					m, n,
					mtk::mateval::col_major,
					mtk::mateval::col_major,
					c_ptr, m,
					reinterpret_cast<C_T*>(mat_ref_uptr.get()), m
					);
		}

		std::printf("%s,%lu,%lu,%lu,%e,%e\n",
				mtk::oztcecgemm::get_compute_mode_name_str(mode).c_str(),
				m, n, k,
				error.at(mtk::mateval::relative_residual),
				error.at(mtk::mateval::max_relative_error)
				);
		std::fflush(stdout);
	}

	mtk::oztcecgemm::destroy(oztcecgemm_handle);
}

std::vector<mtk::oztcecgemm::compute_mode_t> get_supported_compute_mode() {
	return std::vector<mtk::oztcecgemm::compute_mode_t>{
		mtk::oztcecgemm::sgemm,
		mtk::oztcecgemm::dgemm,
		mtk::oztcecgemm::fp32_split_3,
		mtk::oztcecgemm::fp64_int8_6,
		mtk::oztcecgemm::fp64_int8_7,
		mtk::oztcecgemm::fp64_int8_8,
		mtk::oztcecgemm::fp64_int8_9,
	};
}

std::vector<mtk::oztcecgemm::compute_mode_t> get_compute_mode_list_from_argv(
		const std::size_t count,
		char** argv
		) {
	std::vector<mtk::oztcecgemm::compute_mode_t> mode_list;

	for (std::size_t i = 0; i < count; i++) {
		bool added = false;
		for (const auto m : get_supported_compute_mode()) {
			if (std::string(argv[i]) == mtk::oztcecgemm::get_compute_mode_name_str(m)) {
				mode_list.push_back(m);
				added = true;
				break;
			}
		}
		if (!added) {
			std::fprintf(stderr, "Warning: Unknown compute mode \"%s\"\n", argv[i]);
		}
	}

	return mode_list;
}

void print_usage(
		const char* const program_name
		) {
	std::string compute_mode_list_str = "";
	for (const auto& name : get_supported_compute_mode()) {
		compute_mode_list_str += mtk::oztcecgemm::get_compute_mode_name_str(name) + " ";
	}

	std::printf(
			"Usage:\n"
			"%s matfile [/path/to/A.matrix] [/path/to/B.matrix] [/path/to/Ref.matrix] [Computing mode list]\n"
			"%s [urand01|normal01] [seq|exp2] [start_N] [end_N] [interval_N] [Computing mode list]\n"
			"Compute modes:\n"
			" %s\n",
			program_name,
			program_name,
			compute_mode_list_str.c_str()
			);
}

int main(int argc, char** argv) {

	if (argc <= 2) {
		print_usage(argv[0]);
		return 1;
	}

	const auto input_mode = std::string(argv[1]);
	if (input_mode == "matfile") {
		if (argc <= 5) {
			print_usage(argv[0]);
			return 1;
		}

		const auto matfile_A_path = std::string(argv[2]);
		const auto matfile_B_path = std::string(argv[3]);
		const auto matfile_C_path = std::string(argv[4]);
		const auto compute_mode_list = get_compute_mode_list_from_argv(argc - 5, argv + 5);

		std::size_t am, an, bm, bn, cm, cn;
		mtk::matfile::load_size(am, an, matfile_A_path);
		mtk::matfile::load_size(bm, bn, matfile_B_path);
		mtk::matfile::load_size(cm, cn, matfile_C_path);
		if (am != cm || bn != cn || an != bm) {
			std::fprintf(stderr, "Error: matrix shapes are mismatch: A=(%lu, %lu), B=(%lu, %lu), C=(%lu, %lu)\n",
					am, an,
					bm, bn,
					cm, cn
					);
			return 1;
		}

		mtk::oztcecgemm::gemm_list_t fp32in_gemm_list;
		mtk::oztcecgemm::gemm_list_t fp64in_gemm_list;

		for (auto compute_mode : compute_mode_list) {
			if (mtk::oztcecgemm::get_output_type(compute_mode) == mtk::oztcecgemm::fp32) {
				fp32in_gemm_list.push_back(std::tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>(
							cm,
							cn,
							an,
							compute_mode
							));
			} else {
				fp64in_gemm_list.push_back(std::tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>(
							cm,
							cn,
							an,
							compute_mode
							));
			}
		}

		std::printf("mode,m,n,k,residual,max_relative,throughput_in_tflops\n");
		std::fflush(stdout);
		if (fp32in_gemm_list.size() != 0) {
			gemm_eval<float>(fp32in_gemm_list, input_mode);
		}
		if (fp64in_gemm_list.size() != 0) {
			gemm_eval<double>(fp64in_gemm_list, input_mode);
		}

	} else if (input_mode == "urand01" || input_mode == "normal01") {
		if (argc <= 6) {
			print_usage(argv[0]);
			return 1;
		}
		const auto N_mode = std::string(argv[2]);
		if (N_mode != "seq" && N_mode != "exp2") {
			std::fprintf(stderr, "Error: unknown N mode \"%s\"\n", N_mode.c_str());
			return 1;
		}
		const auto min_N = std::stoul(argv[3]);
		const auto max_N = std::stoul(argv[4]);
		const auto interval_N = std::stoul(argv[5]);
		const auto compute_mode_list = get_compute_mode_list_from_argv(argc - 6, argv + 6);

		mtk::oztcecgemm::gemm_list_t fp32in_gemm_list;
		mtk::oztcecgemm::gemm_list_t fp64in_gemm_list;

		for (std::size_t N = min_N; N <= max_N; N += interval_N) {
			auto real_N = N;
			if (N_mode == "exp2") {real_N = 1lu << N;}

			for (auto compute_mode : compute_mode_list) {
				if (mtk::oztcecgemm::get_output_type(compute_mode) == mtk::oztcecgemm::fp32) {
					fp32in_gemm_list.push_back(std::tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>(
								real_N,
								real_N,
								real_N,
								compute_mode
								));
				} else {
					fp64in_gemm_list.push_back(std::tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>(
								real_N,
								real_N,
								real_N,
								compute_mode
								));
				}
			}
		}

		std::printf("mode,m,n,k,residual,max_relative,throughput_in_tflops\n");
		std::fflush(stdout);
		if (fp32in_gemm_list.size() != 0) {
			gemm_eval<float>(fp32in_gemm_list, input_mode);
		}
		if (fp64in_gemm_list.size() != 0) {
			gemm_eval<double>(fp64in_gemm_list, input_mode);
		}
	} else {
		std::fprintf(stderr, "Error: Unknown input mode \"%s\"\n", input_mode.c_str());
		return 1;
	}
}
