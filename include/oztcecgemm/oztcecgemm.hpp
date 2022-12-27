#pragma once
#include <cstdint>

namespace mtk {
namespace oztcecgemm {
struct handle_t;
enum operation_t {
	op_n,
	op_t
};

enum compute_mode_t {
};

int create(handle_t& handle);
int destroy(handle_t& handle);
void set_cuda_stream(handle_t& handle, const cudaStream_t cuda_stream);

template <class AB_T, class C_T>
int gemm(
		const oztcecgemm::handle_t& handle,
		const oztcecgemm::operation_t op_A,
		const oztcecgemm::operation_t op_B,
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const C_T alpha,
		const AB_T* const a_ptr, const std::size_t lda,
		const AB_T* const b_ptr, const std::size_t ldb,
		const C_T beta,
		C_T* const c_ptr, std::size_t ldc,
		const oztcecgemm::compute_mode_t compute_mode
		);
} // namespace oztcecgemm
} // namespace mtk
