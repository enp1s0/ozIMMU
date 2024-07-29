#pragma once
#include <ozimmu/ozimmu.hpp>
#include <vector>

namespace mtk {
namespace ozimmu {
namespace detail {

enum matrix_t { matrix_A, matrix_B, matrix_C };

enum gemm_t {
  cublas_dgemm,
  cublas_sgemm,
  cublas_tf32,
  cublas_fp16,
  cublas_bf16,
  int8tc
};

template <class T> inline data_t get_data_t();
template <> inline data_t get_data_t<float>() { return data_t::fp32; }

struct gemm_pair_config_t {
  // '-1' means the original matrix
  int A_id, B_id;
  gemm_t gemm_mode;
};

struct split_config_t {
  // {[0] = original_type, [1] = ...}
  std::vector<data_t> matrix_A_split_types;
  std::vector<data_t> matrix_B_split_types;
  std::vector<gemm_pair_config_t> gemm_pair_config_list;
};

split_config_t get_split_config(const mtk::ozimmu::compute_mode_t compute_mode);

std::string gemm_mode_str(const gemm_t gemm_mode);

std::size_t
calculate_working_memory_size(const mtk::ozimmu::operation_t op,
                              const std::size_t m, const std::size_t n,
                              const mtk::ozimmu::compute_mode_t compute_mode,
                              const mtk::ozimmu::detail::matrix_t matrix,
                              const mtk::ozimmu::element_kind_t element_kind);

} // namespace detail
} // namespace ozimmu
} // namespace mtk
