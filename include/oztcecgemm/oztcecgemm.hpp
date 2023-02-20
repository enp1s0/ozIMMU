#pragma once
#include <vector>
#include <tuple>
#include <string>

namespace mtk {
namespace oztcecgemm {
struct handle;

using handle_t = handle*;
enum operation_t {
	op_n,
	op_t
};

enum compute_mode_t {
	sgemm,
	dgemm,

	fp32_split_3,

	fp64_int8_6,
	fp64_int8_7,
	fp64_int8_8,
	fp64_int8_9,
	fp64_int8_10,
	fp64_int8_11,
	fp64_int8_12,
	fp64_int8_13,
};

enum data_t {
	fp64,
	fp32,
	fp16,
	int8,
	original
};

int create (mtk::oztcecgemm::handle_t* handle);
int destroy(mtk::oztcecgemm::handle_t handle);
void set_cuda_stream(mtk::oztcecgemm::handle_t handle, const cudaStream_t cuda_stream);

void enable_profiling(mtk::oztcecgemm::handle_t handle);
void disable_profiling(mtk::oztcecgemm::handle_t handle);
void print_profiler_result(mtk::oztcecgemm::handle_t handle, const bool csv = false);
void clear_profiler_result(mtk::oztcecgemm::handle_t handle);

using gemm_list_t = std::vector<std::tuple<std::size_t, std::size_t, std::size_t, mtk::oztcecgemm::compute_mode_t>>;

void reallocate_working_memory(
		mtk::oztcecgemm::handle_t handle,
		const gemm_list_t gemm_list
		);

int gemm(
		mtk::oztcecgemm::handle_t handle,
		const mtk::oztcecgemm::operation_t op_A,
		const mtk::oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const void* alpha,
		const void* const a_ptr, const std::size_t lda,
		const void* const b_ptr, const std::size_t ldb,
		const void* beta,
		void* const c_ptr, std::size_t ldc,
		const mtk::oztcecgemm::compute_mode_t compute_mode
		);

std::string get_compute_mode_name_str(
		const mtk::oztcecgemm::compute_mode_t mode
		);

mtk::oztcecgemm::data_t get_output_type(
		const mtk::oztcecgemm::compute_mode_t mode
		);

std::size_t get_data_size_in_byte(
		const mtk::oztcecgemm::data_t d
		);
} // namespace oztcecgemm
} // namespace mtk
