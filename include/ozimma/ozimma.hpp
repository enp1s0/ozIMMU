#pragma once
#include <vector>
#include <tuple>
#include <string>

namespace mtk {
namespace ozimma {
struct handle;

using handle_t = handle*;
enum operation_t {
	op_n,
	op_t
};

enum compute_mode_t {
	sgemm,
	dgemm,

	fp64_int8_6,
	fp64_int8_7,
	fp64_int8_8,
	fp64_int8_9,
	fp64_int8_10,
	fp64_int8_11,
	fp64_int8_12,
	fp64_int8_13,

	fp64_int8_auto,
};

enum data_t {
	fp64,
	fp32,
	fp16,
	int8,
	original
};

enum malloc_mode_t {
	malloc_sync,
	malloc_async
};

enum element_kind_t {
	real,
	complx,
};

int create (mtk::ozimma::handle_t* handle, const malloc_mode_t mm = malloc_async);
int destroy(mtk::ozimma::handle_t handle);
void set_cuda_stream(mtk::ozimma::handle_t handle, const cudaStream_t cuda_stream);

void enable_profiling(mtk::ozimma::handle_t handle);
void disable_profiling(mtk::ozimma::handle_t handle);
void print_profiler_result(mtk::ozimma::handle_t handle, const bool csv = false);
void clear_profiler_result(mtk::ozimma::handle_t handle);

using gemm_list_t = std::vector<std::tuple<std::size_t, std::size_t, std::size_t, mtk::ozimma::element_kind_t, mtk::ozimma::compute_mode_t>>;

// ReturnA: memory size if reallocated; otherwise, zero
std::size_t reallocate_working_memory(
		mtk::ozimma::handle_t handle,
		const gemm_list_t gemm_list
		);

int gemm(
		mtk::ozimma::handle_t handle,
		const mtk::ozimma::operation_t op_A,
		const mtk::ozimma::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* alpha,
		const void* const a_ptr, const std::size_t lda,
		const void* const b_ptr, const std::size_t ldb,
		const void* beta,
		void* const c_ptr, std::size_t ldc,
		const mtk::ozimma::compute_mode_t compute_mode,
		const mtk::ozimma::element_kind_t element_kind
		);

compute_mode_t auto_mode_select(
		mtk::ozimma::handle_t handle,
		const mtk::ozimma::operation_t op_A,
		const mtk::ozimma::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* const a_ptr, const std::size_t lda,
		const void* const b_ptr, const std::size_t ldb,
		const mtk::ozimma::element_kind_t element_kind,
		const double mantissa_loss_threshold
		);

std::string get_compute_mode_name_str(
		const mtk::ozimma::compute_mode_t mode
		);

mtk::ozimma::data_t get_output_type(
		const mtk::ozimma::compute_mode_t mode
		);

std::size_t get_data_size_in_byte(
		const mtk::ozimma::data_t d
		);
} // namespace ozimma
} // namespace mtk
