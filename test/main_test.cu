#include <algorithm>
#include <chrono>
#include <cuComplex.h>
#include <cutf/curand.hpp>
#include <cutf/curand_kernel.hpp>
#include <cutf/device.hpp>
#include <cutf/math.hpp>
#include <cutf/memory.hpp>
#include <gpu_monitor/gpu_monitor.hpp>
#include <iostream>
#include <mateval/cuda/comparison.hpp>
#include <matfile/matfile.hpp>
#include <ozimmu/ozimmu.hpp>

constexpr unsigned long long seed = 0;

std::string get_gpu_name_str() {
  const auto device_prop = cutf::device::get_properties_vector()[0];
  std::string gpu_name = device_prop.name;
  std::replace(gpu_name.begin(), gpu_name.end(), ' ', '_');
  return gpu_name;
}

inline mtk::mateval::layout_t
conv_layout_oz2mateval(const mtk::ozimmu::operation_t op) {
  if (op == mtk::ozimmu::op_n) {
    return mtk::mateval::col_major;
  }
  return mtk::mateval::row_major;
}

template <class T>
__global__ void adjust_urand_kernel(T *const ptr, const T min_urand,
                                    const T max_urand, const std::size_t n) {
  const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n) {
    return;
  }

  const auto v = ptr[tid];
  ptr[tid] = v * (max_urand - min_urand) + min_urand;
}

template <class T>
void adjust_urand(T *const ptr, const T min_urand, const T max_urand,
                  const std::size_t n) {
  const auto block_size = 256lu;
  const auto grid_size = (n + block_size - 1) / block_size;

  adjust_urand_kernel<T>
      <<<grid_size, block_size>>>(ptr, min_urand, max_urand, n);
}

// See "5.1 Experimental Settings" of
// https://link.springer.com/chapter/10.1007/978-3-030-50743-5_12
template <class T>
__global__ void gen_exp_rand_kernel(T *const ptr, const std::size_t N,
                                    const T phi, const std::uint64_t seed) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  curandStateXORWOW_t curand_state;
  curand_init(seed, tid, 0, &curand_state);

  for (std::size_t i = tid; i < N; i += blockDim.x * gridDim.x) {
    const auto rand = cutf::curand_kernel::uniform<T>(&curand_state);
    const auto randn = cutf::curand_kernel::normal<T>(&curand_state);
    const auto v = (rand - static_cast<T>(0.5)) * exp(phi * randn);

    ptr[i] = v;
  }
}

template <class T>
void gen_exp_rand(T *const ptr, const std::size_t N, const T phi,
                  const std::uint64_t seed) {
  const std::size_t block_size = 256;
  const std::size_t grid_size =
      std::min((N + block_size - 1) / block_size, 1024lu);

  gen_exp_rand_kernel<<<grid_size, block_size>>>(ptr, N, phi, seed);
}

template <class AB_T, class C_T, class MATMUL_FUNC>
int gemm_eval_core(
    const mtk::ozimmu::operation_t op_a, const mtk::ozimmu::operation_t op_b,
    const std::size_t m, const std::size_t n, const std::size_t k,
    const AB_T *const a_ptr, const std::size_t lda, const AB_T *const b_ptr,
    const std::size_t ldb, C_T *const c_ptr, const std::size_t ldc,
    const MATMUL_FUNC matmul_func, const mtk::ozimmu::compute_mode_t mode,
    const mtk::ozimmu::element_kind_t element_kind,
    const std::string input_mode, const std::uint32_t test_count = 100,
    const double error_threshold = 0) {
  int error_flag = 0;

  error_flag =
      matmul_func(op_a, op_b, m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);

  if (error_flag) {
    return 1;
  }

  mtk::mateval::error_map_t error;
  if (element_kind == mtk::ozimmu::real) {
    error = mtk::mateval::cuda::get_error_AxB(
        mtk::mateval::relative_residual | mtk::mateval::max_relative_error, m,
        n, k, conv_layout_oz2mateval(op_a), conv_layout_oz2mateval(op_b),
        mtk::mateval::col_major, reinterpret_cast<const double *>(a_ptr), lda,
        reinterpret_cast<const double *>(b_ptr), ldb,
        reinterpret_cast<const double *>(c_ptr), ldc);
  } else {
    error = mtk::mateval::cuda::get_error_AxB(
        mtk::mateval::relative_residual | mtk::mateval::max_relative_error, m,
        n, k, conv_layout_oz2mateval(op_a), conv_layout_oz2mateval(op_b),
        mtk::mateval::col_major,
        reinterpret_cast<const cuDoubleComplex *>(a_ptr), lda,
        reinterpret_cast<const cuDoubleComplex *>(b_ptr), ldb,
        reinterpret_cast<const cuDoubleComplex *>(c_ptr), ldc);
  }

  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  const auto start_clock = std::chrono::system_clock::now();

  for (unsigned i = 0; i < test_count; i++) {
    error_flag |=
        matmul_func(op_a, op_b, m, n, k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
  }

  if (error_flag) {
    return 1;
  }

  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  const auto end_clock = std::chrono::system_clock::now();

  const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock -
                                                           start_clock)
          .count() *
      1e-9 / test_count;

  const auto throughput = 2 * m * n * k / elapsed_time *
                          (element_kind == mtk::ozimmu::real ? 1 : 4);

  std::printf(
      "%s,%s,%s,%s,%s,%s,%lu,%lu,%lu,%e,%e,%e\n", get_gpu_name_str().c_str(),
      (element_kind == mtk::ozimmu::real ? "D" : "Z"), input_mode.c_str(),
      mtk::ozimmu::get_compute_mode_name_str(mode).c_str(),
      (op_a == mtk::ozimmu::op_n ? "N" : "T"),
      (op_b == mtk::ozimmu::op_n ? "N" : "T"), m, n, k,
      error.at(mtk::mateval::relative_residual),
      error.at(mtk::mateval::max_relative_error), throughput * 1e-12);
  std::fflush(stdout);

  if (error_threshold != 0) {
    if (error.at(mtk::mateval::relative_residual) < error_threshold) {
      return 0;
    }
    std::printf("^^^ FAILED ^^^^\n");
    std::fflush(stdout);
    return 1;
  }

  return 0;
}

template <class T>
int gemm_eval(const mtk::ozimmu::gemm_list_t &gemm_list,
              const std::string input_mode,
              const std::uint32_t test_count = 100,
              const double error_threshold = 0.0) {
  mtk::ozimmu::handle_t ozimmu_handle;
  mtk::ozimmu::create(&ozimmu_handle);
  mtk::ozimmu::reallocate_working_memory(ozimmu_handle, gemm_list);

  std::size_t max_AB_count = 0;
  std::size_t max_C_size = 0;
  for (const auto gemm : gemm_list) {
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    max_AB_count =
        std::max(max_AB_count,
                 (m * k + k * n) * (element_kind == mtk::ozimmu::real ? 1 : 2));
    max_C_size = std::max(
        max_C_size, m * n *
                        mtk::ozimmu::get_data_size_in_byte(
                            mtk::ozimmu::get_output_type(std::get<6>(gemm))) *
                        (element_kind == mtk::ozimmu::real ? 1 : 2));
  }

  auto mat_AB_uptr = cutf::memory::get_device_unique_ptr<T>(max_AB_count);
  auto mat_C_uptr =
      cutf::memory::get_device_unique_ptr<std::uint8_t>(max_C_size);

  auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
  if (input_mode == "normal01") {
    CUTF_CHECK_ERROR(cutf::curand::generate_normal(
        *cugen.get(), mat_AB_uptr.get(), max_AB_count, 0, 1));
  } else if (input_mode == "urand01") {
    CUTF_CHECK_ERROR(cutf::curand::generate_uniform(
        *cugen.get(), mat_AB_uptr.get(), max_AB_count));
  } else {
    double phi = 0;
    try {
      phi = std::stod(input_mode.substr(9));
    } catch (const std::exception &e) {
      std::fprintf(stderr, "Error: %s [%s (line:%d)]\n", e.what(), __FILE__,
                   __LINE__);
      return 1;
    }
    gen_exp_rand<T>(mat_AB_uptr.get(), max_AB_count, phi, 0);
  }

  std::uint32_t num_errors = 0;
  for (const auto gemm : gemm_list) {
    const auto op_A = std::get<0>(gemm);
    const auto op_B = std::get<1>(gemm);
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    const auto mode = std::get<6>(gemm);

    const auto lda_r = op_A == mtk::ozimmu::op_n ? m : k;
    const auto ldb_r = op_B == mtk::ozimmu::op_n ? k : n;

    T *const a_ptr = mat_AB_uptr.get();
    T *const b_ptr =
        a_ptr + m * k * (element_kind == mtk::ozimmu::real ? 1 : 2);

    num_errors += gemm_eval_core(
        op_A, op_B, m, n, k, a_ptr, lda_r, b_ptr, ldb_r, mat_C_uptr.get(), m,
        [&](const mtk::ozimmu::operation_t op_a,
            const mtk::ozimmu::operation_t op_b, const std::size_t m,
            const std::size_t n, const std::size_t k, const T *const a_ptr,
            const std::size_t lda, const T *const b_ptr, const std::size_t ldb,
            void *const c_ptr, const std::size_t ldc) -> int {
          if (element_kind == mtk::ozimmu::real) {
            using C_T = double;
            const C_T alpha = 1, beta = 0;
            return mtk::ozimmu::gemm(ozimmu_handle, op_a, op_b, m, n, k, &alpha,
                                     a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc,
                                     mode, element_kind);
          } else {
            using C_T = cuDoubleComplex;
            const C_T alpha = make_cuDoubleComplex(1, 0),
                      beta = make_cuDoubleComplex(0, 0);
            return mtk::ozimmu::gemm(ozimmu_handle, op_a, op_b, m, n, k, &alpha,
                                     a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc,
                                     mode, element_kind);
          }
        },
        mode, element_kind, input_mode, test_count, error_threshold);
  }

  mtk::ozimmu::destroy(ozimmu_handle);
  return num_errors;
}

template <class SRC_T, class DST_T>
__global__ void vector_copy_kernel(DST_T *const dst_ptr,
                                   const SRC_T *const src_ptr,
                                   const std::size_t N) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) {
    return;
  }

  dst_ptr[tid] = src_ptr[tid];
}

template <class DEVICE_T>
void matfile_to_device_memory(DEVICE_T *const d_ptr,
                              const std::string matfile_path) {
  const auto [m, n] = mtk::matfile::load_matrix_size(matfile_path);
  const auto dtype = mtk::matfile::load_dtype(matfile_path);

  auto h_mat_uptr = cutf::memory::get_host_unique_ptr<std::uint8_t>(
      m * n * mtk::matfile::get_dtype_size(dtype));

  const std::size_t block_size = 256;
  const std::size_t grid_size = (m * n + block_size - 1) / block_size;

  if (dtype == mtk::matfile::data_t::fp32) {
    mtk::matfile::load_dense(reinterpret_cast<float *>(h_mat_uptr.get()), m,
                             matfile_path);
    vector_copy_kernel<<<grid_size, block_size>>>(
        d_ptr, reinterpret_cast<float *>(h_mat_uptr.get()), m * n);
  } else if (dtype == mtk::matfile::data_t::fp64) {
    mtk::matfile::load_dense(reinterpret_cast<double *>(h_mat_uptr.get()), m,
                             matfile_path);
    vector_copy_kernel<<<grid_size, block_size>>>(
        d_ptr, reinterpret_cast<double *>(h_mat_uptr.get()), m * n);
  } else {
    mtk::matfile::load_dense(reinterpret_cast<long double *>(h_mat_uptr.get()),
                             m, matfile_path);
    vector_copy_kernel<<<grid_size, block_size>>>(
        d_ptr, reinterpret_cast<long double *>(h_mat_uptr.get()), m * n);
  }
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
}

template <class C_T>
mtk::mateval::error_map_t eval_matfile(const std::string matfile_C_path,
                                       const C_T *const c_ptr) {
  const auto [m, n] = mtk::matfile::load_matrix_size(matfile_C_path);
  const auto dtype = mtk::matfile::load_dtype(matfile_C_path);

  mtk::mateval::error_map_t error;
  if (dtype == mtk::matfile::data_t::fp32) {
    using R_T = float;
    auto mat_ref_uptr = cutf::memory::get_host_unique_ptr<R_T>(m * n);
    mtk::matfile::load_dense(mat_ref_uptr.get(), m, matfile_C_path);

    error = mtk::mateval::cuda::get_error(
        mtk::mateval::max_relative_error | mtk::mateval::relative_residual, m,
        n, mtk::mateval::col_major, mtk::mateval::col_major, c_ptr, m,
        reinterpret_cast<R_T *>(mat_ref_uptr.get()), m);
  } else if (dtype == mtk::matfile::data_t::fp64) {
    using R_T = double;
    auto mat_ref_uptr = cutf::memory::get_host_unique_ptr<R_T>(m * n);
    mtk::matfile::load_dense(mat_ref_uptr.get(), m, matfile_C_path);

    error = mtk::mateval::cuda::get_error(
        mtk::mateval::max_relative_error | mtk::mateval::relative_residual, m,
        n, mtk::mateval::col_major, mtk::mateval::col_major, c_ptr, m,
        reinterpret_cast<R_T *>(mat_ref_uptr.get()), m);
  }

  return error;
}

template <class T>
void gemm_eval_matfile(const mtk::ozimmu::gemm_list_t &gemm_list,
                       const std::string matfile_A_path,
                       const std::string matfile_B_path) {
  mtk::ozimmu::handle_t ozimmu_handle;
  mtk::ozimmu::create(&ozimmu_handle);
  mtk::ozimmu::reallocate_working_memory(ozimmu_handle, gemm_list);

  std::size_t max_AB_count = 0;
  std::size_t max_C_size = 0;
  for (const auto gemm : gemm_list) {
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    max_AB_count =
        std::max(max_AB_count,
                 (m * k + k * n) * (element_kind == mtk::ozimmu::real ? 1 : 2));
    max_C_size = std::max(
        max_C_size, m * n *
                        mtk::ozimmu::get_data_size_in_byte(
                            mtk::ozimmu::get_output_type(std::get<6>(gemm))) *
                        (element_kind == mtk::ozimmu::real ? 1 : 2));
  }

  auto mat_AB_uptr = cutf::memory::get_device_unique_ptr<T>(max_AB_count);
  auto mat_C_uptr =
      cutf::memory::get_device_unique_ptr<std::uint8_t>(max_C_size);

  for (const auto gemm : gemm_list) {
    const auto op_A = std::get<0>(gemm);
    const auto op_B = std::get<1>(gemm);
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    const auto mode = std::get<6>(gemm);

    const auto a_ptr = mat_AB_uptr.get();
    const auto b_ptr = mat_AB_uptr.get() + m * k;
    const auto c_ptr = mat_C_uptr.get();

    matfile_to_device_memory(a_ptr, matfile_A_path);
    matfile_to_device_memory(b_ptr, matfile_B_path);

    gemm_eval_core(
        op_A, op_B, m, n, k, mat_AB_uptr.get(), m, mat_AB_uptr.get() + m * k, k,
        mat_C_uptr.get(), m,
        [&](const mtk::ozimmu::operation_t op_a,
            const mtk::ozimmu::operation_t op_b, const std::size_t m,
            const std::size_t n, const std::size_t k, const T *const a_ptr,
            const std::size_t lda, const T *const b_ptr, const std::size_t ldb,
            void *const c_ptr, const std::size_t ldc) -> int {
          if (element_kind == mtk::ozimmu::real) {
            using C_T = double;
            const C_T alpha = 1, beta = 0;
            return mtk::ozimmu::gemm(ozimmu_handle, op_a, op_b, m, n, k, &alpha,
                                     a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc,
                                     mode, element_kind);
          } else {
            using C_T = cuDoubleComplex;
            const C_T alpha = make_cuDoubleComplex(1, 0),
                      beta = make_cuDoubleComplex(0, 0);
            return mtk::ozimmu::gemm(ozimmu_handle, op_a, op_b, m, n, k, &alpha,
                                     a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc,
                                     mode, element_kind);
          }
        },
        mode, element_kind, "matfile");
  }

  mtk::ozimmu::destroy(ozimmu_handle);
}

template <class T>
void gemm_eval_power(const mtk::ozimmu::gemm_list_t &gemm_list) {
  mtk::ozimmu::handle_t ozimmu_handle;
  mtk::ozimmu::create(&ozimmu_handle);
  mtk::ozimmu::reallocate_working_memory(ozimmu_handle, gemm_list);

  std::size_t max_AB_count = 0;
  std::size_t max_C_size = 0;
  for (const auto gemm : gemm_list) {
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    max_AB_count =
        std::max(max_AB_count,
                 (m * k + k * n) * (element_kind == mtk::ozimmu::real ? 1 : 2));
    max_C_size = std::max(
        max_C_size, m * n *
                        mtk::ozimmu::get_data_size_in_byte(
                            mtk::ozimmu::get_output_type(std::get<6>(gemm))) *
                        (element_kind == mtk::ozimmu::real ? 1 : 2));
  }

  auto mat_AB_uptr = cutf::memory::get_device_unique_ptr<T>(max_AB_count);
  auto mat_C_uptr =
      cutf::memory::get_device_unique_ptr<std::uint8_t>(max_C_size);

  auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));

  CUTF_CHECK_ERROR(cutf::curand::generate_normal(
      *cugen.get(), mat_AB_uptr.get(), max_AB_count, 0, 1));

  for (const auto gemm : gemm_list) {
    const auto op_A = std::get<0>(gemm);
    const auto op_B = std::get<1>(gemm);
    const auto m = std::get<2>(gemm);
    const auto n = std::get<3>(gemm);
    const auto k = std::get<4>(gemm);
    const auto element_kind = std::get<5>(gemm);
    const auto mode = std::get<6>(gemm);

    const auto gemm_func =
        [&](const mtk::ozimmu::operation_t op_a,
            const mtk::ozimmu::operation_t op_b, const std::size_t m,
            const std::size_t n, const std::size_t k, const T *const a_ptr,
            const std::size_t lda, const T *const b_ptr, const std::size_t ldb,
            void *const c_ptr, const std::size_t ldc) -> int {
      if (mtk::ozimmu::get_output_type(mode) == mtk::ozimmu::fp32) {
        using C_T = float;
        const C_T alpha = 1, beta = 0;
        return mtk::ozimmu::gemm(ozimmu_handle, op_a, op_b, m, n, k, &alpha,
                                 a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc,
                                 mode, element_kind);
      } else {
        using C_T = double;
        const C_T alpha = 1, beta = 0;
        return mtk::ozimmu::gemm(ozimmu_handle, op_a, op_b, m, n, k, &alpha,
                                 a_ptr, lda, b_ptr, ldb, &beta, c_ptr, ldc,
                                 mode, element_kind);
      }
    };

    constexpr std::size_t duration_time = 10;
    std::size_t c = 0;
    const auto result = mtk::gpu_monitor::measure_power_consumption(
        [&]() {
          CUTF_CHECK_ERROR(cudaDeviceSynchronize());
          const auto start_clock = std::chrono::system_clock::now();
          while (true) {
            gemm_func(op_A, op_B, m, n, k, mat_AB_uptr.get(), m,
                      mat_AB_uptr.get() + m * k, k, mat_C_uptr.get(), m);
            if (((++c) % 10) == 0) {
              CUTF_CHECK_ERROR(cudaDeviceSynchronize());
              const auto current_clock = std::chrono::system_clock::now();
              const auto elapsed_time =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      current_clock - start_clock)
                      .count() *
                  1e-6;
              if (elapsed_time > duration_time) {
                break;
              }
            }
          }
        },
        100);
    const auto power =
        mtk::gpu_monitor::get_integrated_power_consumption(result);
    const auto elapsed_time = mtk::gpu_monitor::get_elapsed_time(result);
    const auto average_power = power / elapsed_time;
    const auto flops_per_watt = 2lu * m * n * k * c / power;
    const auto throughput_in_tflops =
        2lu * m * n * k * c / elapsed_time * 1e-12;

    std::printf("%s,%s,%lu,%lu,%lu,%e,%e,%e,%e,%lu\n",
                get_gpu_name_str().c_str(),
                mtk::ozimmu::get_compute_mode_name_str(mode).c_str(), m, n, k,
                throughput_in_tflops, average_power, flops_per_watt * 1e-9,
                elapsed_time, c);
    std::fflush(stdout);
  }

  mtk::ozimmu::destroy(ozimmu_handle);
}

std::vector<mtk::ozimmu::compute_mode_t> get_supported_compute_mode() {
  return std::vector<mtk::ozimmu::compute_mode_t>{
      mtk::ozimmu::sgemm,          mtk::ozimmu::dgemm,
      mtk::ozimmu::fp64_int8_3,    mtk::ozimmu::fp64_int8_4,
      mtk::ozimmu::fp64_int8_5,    mtk::ozimmu::fp64_int8_6,
      mtk::ozimmu::fp64_int8_7,    mtk::ozimmu::fp64_int8_8,
      mtk::ozimmu::fp64_int8_9,    mtk::ozimmu::fp64_int8_10,
      mtk::ozimmu::fp64_int8_11,   mtk::ozimmu::fp64_int8_12,
      mtk::ozimmu::fp64_int8_13,   mtk::ozimmu::fp64_int8_14,
      mtk::ozimmu::fp64_int8_15,   mtk::ozimmu::fp64_int8_16,
      mtk::ozimmu::fp64_int8_17,   mtk::ozimmu::fp64_int8_18,
      mtk::ozimmu::fp64_int8_auto,
  };
}

std::vector<mtk::ozimmu::compute_mode_t>
get_compute_mode_list_from_argv(const std::size_t count, char **argv) {
  std::vector<mtk::ozimmu::compute_mode_t> mode_list;

  for (std::size_t i = 0; i < count; i++) {
    bool added = false;
    for (const auto m : get_supported_compute_mode()) {
      if (std::string(argv[i]) == mtk::ozimmu::get_compute_mode_name_str(m)) {
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

void print_usage(const char *const program_name) {
  std::string compute_mode_list_str = "";
  for (const auto &name : get_supported_compute_mode()) {
    compute_mode_list_str += mtk::ozimmu::get_compute_mode_name_str(name) + " ";
  }

  std::printf("Usage:\n"
              "%s matfile [/path/to/A.matrix] [/path/to/B.matrix] [Computing "
              "mode list]\n"
              "%s [urand01 | normal01 | exp_rand-X] [zgemm | dgemm] [seq|exp2] "
              "[start_N] [end_N] [interval_N] [Computing mode list]\n"
              "%s power [seq|exp2] [start_N] [end_N] [interval_N] [Computing "
              "mode list]\n"
              "%s ci_test\n"
              "Compute modes:\n"
              " %s\n",
              program_name, program_name, program_name, program_name,
              compute_mode_list_str.c_str());
}

int main(int argc, char **argv) {

  if (argc <= 1) {
    print_usage(argv[0]);
    return 1;
  }

  const auto input_mode = std::string(argv[1]);
  if (input_mode == "matfile") {
    if (argc <= 4) {
      print_usage(argv[0]);
      return 1;
    }

    const auto matfile_A_path = std::string(argv[2]);
    const auto matfile_B_path = std::string(argv[3]);
    const auto compute_mode_list =
        get_compute_mode_list_from_argv(argc - 4, argv + 4);

    const auto [am, an] = mtk::matfile::load_matrix_size(matfile_A_path);
    const auto [bm, bn] = mtk::matfile::load_matrix_size(matfile_B_path);
    if (an != bm) {
      std::fprintf(stderr,
                   "Error: matrix shapes are mismatch: A=(%lu, %lu), B=(%lu, "
                   "%lu), C=(%lu, %lu)\n",
                   am, an, bm, bn, am, bn);
      return 1;
    }

    mtk::ozimmu::gemm_list_t fp64in_gemm_list;

    for (auto compute_mode : compute_mode_list) {
      fp64in_gemm_list.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n, am, bn,
                                  an, mtk::ozimmu::real, compute_mode});
    }

    std::printf("matfile test:\n"
                "A : %s\n"
                "B : %s\n",
                matfile_A_path.c_str(), matfile_B_path.c_str());
    std::printf("gpu,gemm,input,mode,m,n,k,residual,max_relative\n");
    std::fflush(stdout);
    if (fp64in_gemm_list.size() != 0) {
      gemm_eval_matfile<double>(fp64in_gemm_list, matfile_A_path,
                                matfile_B_path);
    }
  } else if (input_mode == "urand01" || input_mode == "normal01" ||
             (input_mode.length() >= 9 &&
              input_mode.substr(0, 9) == "exp_rand-")) {
    if (argc <= 7) {
      print_usage(argv[0]);
      return 1;
    }

    const auto gemm = std::string(argv[2]);
    if (gemm != "dgemm" && gemm != "zgemm") {
      std::fprintf(stderr, "Error: unknown gemm \"%s\"\n", gemm.c_str());
      return 1;
    }

    const auto N_mode = std::string(argv[3]);
    if (N_mode != "seq" && N_mode != "exp2") {
      std::fprintf(stderr, "Error: unknown N mode \"%s\"\n", N_mode.c_str());
      return 1;
    }
    const auto min_N = std::stoul(argv[4]);
    const auto max_N = std::stoul(argv[5]);
    const auto interval_N = std::stoul(argv[6]);
    const auto compute_mode_list =
        get_compute_mode_list_from_argv(argc - 7, argv + 7);

    mtk::ozimmu::gemm_list_t fp32in_gemm_list;
    mtk::ozimmu::gemm_list_t fp64in_gemm_list;

    for (std::size_t N = min_N; N <= max_N; N += interval_N) {
      auto real_N = N;
      if (N_mode == "exp2") {
        real_N = 1lu << N;
      }

      for (auto compute_mode : compute_mode_list) {
        fp64in_gemm_list.push_back(
            {mtk::ozimmu::op_n, mtk::ozimmu::op_n, real_N, real_N, real_N,
             gemm == "dgemm" ? mtk::ozimmu::real : mtk::ozimmu::complx,
             compute_mode});
      }
    }

    std::printf("gpu,gemm,input,mode,m,n,k,residual,max_relative,throughput_in_"
                "tflops\n");
    std::fflush(stdout);
    if (fp64in_gemm_list.size() != 0) {
      gemm_eval<double>(fp64in_gemm_list, input_mode);
    }
  } else if (input_mode == "power") {
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
    const auto compute_mode_list =
        get_compute_mode_list_from_argv(argc - 6, argv + 6);

    mtk::ozimmu::gemm_list_t fp32in_gemm_list;
    mtk::ozimmu::gemm_list_t fp64in_gemm_list;

    for (std::size_t N = min_N; N <= max_N; N += interval_N) {
      auto real_N = N;
      if (N_mode == "exp2") {
        real_N = 1lu << N;
      }

      for (auto compute_mode : compute_mode_list) {
        fp64in_gemm_list.push_back({mtk::ozimmu::op_n, mtk::ozimmu::op_n,
                                    real_N, real_N, real_N, mtk::ozimmu::real,
                                    compute_mode});
      }
    }

    std::printf("gpu,mode,m,n,k,throughput_in_tflops,avg_watt,gflops_per_watt,"
                "time,count\n");
    std::fflush(stdout);
    if (fp64in_gemm_list.size() != 0) {
      gemm_eval_power<double>(fp64in_gemm_list);
    }
  } else if (input_mode == "ci_test") {
    std::vector<std::size_t> n_list = {1023, 1024, 1025};
    std::vector<mtk::ozimmu::operation_t> op_list = {mtk::ozimmu::op_n,
                                                     mtk::ozimmu::op_t};
    std::vector<mtk::ozimmu::compute_mode_t> mode_list = {
        mtk::ozimmu::fp64_int8_8,  mtk::ozimmu::fp64_int8_9,
        mtk::ozimmu::fp64_int8_10, mtk::ozimmu::fp64_int8_11,
        mtk::ozimmu::fp64_int8_12, mtk::ozimmu::fp64_int8_13,
        mtk::ozimmu::fp64_int8_14, mtk::ozimmu::fp64_int8_15,
        mtk::ozimmu::fp64_int8_16};

    mtk::ozimmu::gemm_list_t gemm_list;
    // Real
    for (const auto &op_A : op_list) {
      for (const auto &op_B : op_list) {
        for (const auto &m : n_list) {
          for (const auto &n : n_list) {
            for (const auto &k : n_list) {
              for (const auto &compute_mode : mode_list) {
                gemm_list.push_back(
                    {op_A, op_B, m, n, k, mtk::ozimmu::real, compute_mode});
              }
            }
          }
        }
      }
    }
    // Complex
    for (const auto &op_A : op_list) {
      for (const auto &op_B : op_list) {
        for (const auto &m : n_list) {
          for (const auto &n : n_list) {
            for (const auto &k : n_list) {
              for (const auto &compute_mode : mode_list) {
                gemm_list.push_back(
                    {op_A, op_B, m, n, k, mtk::ozimmu::complx, compute_mode});
              }
            }
          }
        }
      }
    }
    const auto num_errors = gemm_eval<double>(gemm_list, "urand01", 1, 1e-15);
    std::printf("%5lu / %5lu PASSED\n", gemm_list.size() - num_errors,
                gemm_list.size());
  } else {
    std::fprintf(stderr, "Error: Unknown input mode \"%s\"\n",
                 input_mode.c_str());
    return 1;
  }
}
