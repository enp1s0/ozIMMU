#include "handle.hpp"

int mtk::oztcecgemm::create(
		mtk::oztcecgemm::handle_t& handle
		) {
	// Initialize cuBLAS handler
	CUTF_CHECK_ERROR(cublasCreate(&(handle.cublas_handle)));

	// Initialize SHGEMM handler
	mtk::shgemm::create(handle.shgemm_handle);

	return 0;
}

int mtk::oztcecgemm::destroy(
		mtk::oztcecgemm::handle_t& handle
		) {
	// Destroy cuBLAS handler
	CUTF_CHECK_ERROR(cublasDestroy(handle.cublas_handle));

	// Destroy SHGEMM handler
	mtk::shgemm::destroy(handle.shgemm_handle);

	return 0;
}

void mtk::oztcecgemm::set_cuda_stream(
		mtk::oztcecgemm::handle_t& handle,
		cudaStream_t cuda_stream
		) {
	// Set cuda stream to cuBLAS handler
	CUTF_CHECK_ERROR(cublasSetStream(handle.cublas_handle, cuda_stream));

	// Set cuda stream to SHGEMM handler
	mtk::shgemm::set_cuda_stream(handle.shgemm_handle, cuda_stream);
}
