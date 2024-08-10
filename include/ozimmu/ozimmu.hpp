#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace mtk {
namespace ozimmu {
struct handle;

using handle_t = handle *;
enum operation_t { op_n, op_t };

enum compute_mode_t {
  sgemm,
  dgemm,

  fp64_int8_3,
  fp64_int8_4,
  fp64_int8_5,
  fp64_int8_6,
  fp64_int8_7,
  fp64_int8_8,
  fp64_int8_9,
  fp64_int8_10,
  fp64_int8_11,
  fp64_int8_12,
  fp64_int8_13,
  fp64_int8_14,
  fp64_int8_15,
  fp64_int8_16,
  fp64_int8_17,
  fp64_int8_18,

  fp64_int8_auto,
};

enum data_t { fp64, fp32, fp16, int8, original, none };

enum malloc_mode_t { malloc_sync, malloc_async };

enum element_kind_t {
  real,
  complx,
};

int create(mtk::ozimmu::handle_t *handle, const malloc_mode_t mm = malloc_sync);
int destroy(mtk::ozimmu::handle_t handle);
void set_cuda_stream(mtk::ozimmu::handle_t handle,
                     const cudaStream_t cuda_stream);

void enable_profiling(mtk::ozimmu::handle_t handle);
void disable_profiling(mtk::ozimmu::handle_t handle);
void print_profiler_result(mtk::ozimmu::handle_t handle, const std::string tag,
                           const bool csv = false);
void clear_profiler_result(mtk::ozimmu::handle_t handle);

void set_auto_mantissa_loss_threashold(mtk::ozimmu::handle_t handle,
                                       const double threshold);
double get_auto_mantissa_loss_threashold(mtk::ozimmu::handle_t handle);

using gemm_params_t =
    std::tuple<mtk::ozimmu::operation_t, mtk::ozimmu::operation_t, std::size_t,
               std::size_t, std::size_t, mtk::ozimmu::element_kind_t,
               mtk::ozimmu::compute_mode_t>;
using gemm_list_t = std::vector<gemm_params_t>;

// ReturnA: memory size if reallocated; otherwise, zero
std::size_t reallocate_working_memory(mtk::ozimmu::handle_t handle,
                                      const gemm_list_t gemm_list);

std::size_t reallocate_working_memory(mtk::ozimmu::handle_t handle,
                                      const std::size_t size_in_byte);

int gemm(mtk::ozimmu::handle_t handle, const mtk::ozimmu::operation_t op_A,
         const mtk::ozimmu::operation_t op_B, const std::size_t m,
         const std::size_t n, const std::size_t k, const void *alpha,
         const void *const a_ptr, const std::size_t lda,
         const void *const b_ptr, const std::size_t ldb, const void *beta,
         void *const c_ptr, std::size_t ldc,
         const mtk::ozimmu::compute_mode_t compute_mode,
         const mtk::ozimmu::element_kind_t element_kind);

compute_mode_t auto_mode_select(mtk::ozimmu::handle_t handle,
                                const mtk::ozimmu::operation_t op_A,
                                const mtk::ozimmu::operation_t op_B,
                                const std::size_t m, const std::size_t n,
                                const std::size_t k, const void *const a_ptr,
                                const std::size_t lda, const void *const b_ptr,
                                const std::size_t ldb,
                                const mtk::ozimmu::element_kind_t element_kind,
                                const double mantissa_loss_threshold);

std::string get_compute_mode_name_str(const mtk::ozimmu::compute_mode_t mode);

mtk::ozimmu::data_t get_output_type(const mtk::ozimmu::compute_mode_t mode);

std::size_t get_data_size_in_byte(const mtk::ozimmu::data_t d);

std::uint32_t get_bits_per_int8(const std::uint32_t k);
} // namespace ozimmu
} // namespace mtk
