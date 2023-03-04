#include <unistd.h>
#include <dlfcn.h>
#include <cutf/cublas.hpp>
#include <ozimma/ozimma.hpp>
#include "culip.hpp"
#include "handle.hpp"

#ifndef CUBLASAPI
#define CUBLASAPI
#endif

// For logging
template <class Func>
void ozTCECGEMM_run_if_env_defined(
		const std::string env_str,
		const Func func
		) {
	const auto env = getenv(env_str.c_str());
	if (env != nullptr && std::string(env) != "0") {
		func();
	}
}


const std::string info_env_name = "OZIMMA_INFO";
void ozIMMA_log(
		const std::string str
		) {
	ozTCECGEMM_run_if_env_defined(
			info_env_name,
			[&](){
				std::fprintf(stdout, "[ozIMMA LOG] %s\n",
						str.c_str());
				std::fflush(stdout);
			});
}

const std::string error_env_name = "OZIMMA_ERROR_LOG";
void ozIMMA_error(
		const std::string str
		) {
	ozTCECGEMM_run_if_env_defined(
			error_env_name,
			[&](){
				std::fprintf(stdout, "[ozIMMA ERROR] %s\n",
						str.c_str());
				std::fflush(stdout);
			});
}

void* ozIMMA_get_function_pointer(const std::string library_name, const std::string function_name) {

	// Open the library
	const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
	if (lib_ptr == nullptr) {
		ozIMMA_error("Failed to load " + library_name + ". Default rule will be used.");
		return nullptr;
	}

	// Get function pointer
	void* function_ptr = dlsym(lib_ptr, function_name.c_str());
	if (function_ptr == NULL) {
		ozIMMA_log("Failed to load a function " + function_name + " during selecting hijacking function. Default rule will be used.");
		return nullptr;
	}

	return function_ptr;
}

mtk::ozimma::gemm_list_t get_default_gemm_list() {
	return mtk::ozimma::gemm_list_t{
		std::tuple<std::size_t, std::size_t, std::size_t, mtk::ozimma::compute_mode_t>{1024, 1024, 1024, mtk::ozimma::fp64_int8_9}
	};
}

mtk::ozimma::compute_mode_t get_compute_mode(
		const std::size_t m,
		const std::size_t n,
		const std::size_t k
		) {
	const char* env_name = "OZIMMA_COMPUTE_MODE";
	const char* env_val = getenv(env_name);

	std::vector<mtk::ozimma::compute_mode_t> supported_gemm_mode = {
		mtk::ozimma::sgemm,
		mtk::ozimma::dgemm,
		mtk::ozimma::fp64_int8_6,
		mtk::ozimma::fp64_int8_7,
		mtk::ozimma::fp64_int8_8,
		mtk::ozimma::fp64_int8_9,
		mtk::ozimma::fp64_int8_10,
		mtk::ozimma::fp64_int8_11,
		mtk::ozimma::fp64_int8_12,
		mtk::ozimma::fp64_int8_13,
	};

	if (env_val != nullptr) {
		const std::string env_val_str = env_val;

		for (const auto mode : supported_gemm_mode) {
			if (mtk::ozimma::get_compute_mode_name_str(mode) == env_val_str) {
				return mode;
			}
		}
	}

	return mtk::ozimma::dgemm;
}

mtk::ozimma::operation_t op_cublas2oz(
		const cublasOperation_t op
		) {
	if (op == CUBLAS_OP_N) {
		return mtk::ozimma::op_n;
	} else {
		return mtk::ozimma::op_t;
	}
}

mtk::ozimma::handle_t* global_ozimma_handle = nullptr;

mtk::ozimma::handle_t& get_global_ozimma_handle() {
	if (global_ozimma_handle == nullptr) {
		ozIMMA_log("Initializing ozIMMA handle...");
		global_ozimma_handle = new mtk::ozimma::handle_t;
		mtk::ozimma::create(global_ozimma_handle);
	}
	return *global_ozimma_handle;
}

std::string cublas_library_name = "libcublas.so";

cublasStatus_t mtk::ozimma::cublasCreate_org(
		cublasHandle_t* handle_ptr
		) {
	cublasStatus_t (*func_ptr)(cublasHandle_t*);
	*(void**)(&func_ptr)	= ozIMMA_get_function_pointer(cublas_library_name, "cublasCreate_v2");
	return (*func_ptr)(handle_ptr);
}

cublasStatus_t mtk::ozimma::cublasDestroy_org(
		cublasHandle_t cublas_handle
		) {
	cublasStatus_t (*func_ptr)(cublasHandle_t);
	*(void**)(&func_ptr)	= ozIMMA_get_function_pointer(cublas_library_name, "cublasDestroy_v2");
	return (*func_ptr)(cublas_handle);
}

// Hijacking functions
extern "C" {
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2 (cublasHandle_t *handle) {
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
	// Allocate memory
	const auto reallocated_size = mtk::ozimma::reallocate_working_memory(
			get_global_ozimma_handle(),
			get_default_gemm_list()
			);
	if (reallocated_size != 0) {
		ozIMMA_log("Reallocated moery : " + std::to_string(reallocated_size) + " B");
	}

	// Run the original function
	return mtk::ozimma::cublasCreate_org(handle);
#endif
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2 (cublasHandle_t handle) {
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
	if (global_ozimma_handle != nullptr) {
		ozIMMA_log("Destroying ozIMMA handle...");
		mtk::ozimma::destroy(
				get_global_ozimma_handle()
				);
		delete global_ozimma_handle;
		global_ozimma_handle = nullptr;
	}

	// Run the original function
	return mtk::ozimma::cublasDestroy_org(handle);
#endif
}

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2 (cublasHandle_t handle, 
		cublasOperation_t transa,
		cublasOperation_t transb, 
		int m,
		int n,
		int k,
		const double *alpha,
		const double *A, 
		int lda,
		const double *B,
		int ldb, 
		const double *beta,
		double *C,
		int ldc) {
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
	const auto compute_mode = get_compute_mode(m, n, k);

	if (compute_mode != mtk::ozimma::dgemm) {
		const auto gemm_config = mtk::ozimma::gemm_list_t {
			std::tuple<std::size_t, std::size_t, std::size_t, mtk::ozimma::compute_mode_t>{m, n, k, compute_mode}
		};

		const auto reallocated_size = mtk::ozimma::reallocate_working_memory(
				get_global_ozimma_handle(),
				gemm_config
				);
		if (reallocated_size != 0) {
			ozIMMA_log("Reallocated moery : " + std::to_string(reallocated_size) + " B");
		}

		cudaStream_t cuda_stream;
		CUTF_CHECK_ERROR(cublasGetStream(handle, &cuda_stream));
		mtk::ozimma::set_cuda_stream(get_global_ozimma_handle(), cuda_stream);

		mtk::ozimma::CULiP::profile_result profile_result;
		const auto profiling_flag = mtk::ozimma::CULiP::is_profiling_enabled();

		cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
		*(void**)(&func_ptr) = ozIMMA_get_function_pointer(
				cublas_library_name,
				__func__
				);

		if (profiling_flag) {
			snprintf(profile_result.function_name, profile_result.function_name_length - 1,
					"%s-%s%s-m%d-n%d-k%d",
					mtk::ozimma::get_compute_mode_name_str(compute_mode).c_str(),
					mtk::ozimma::CULiP::get_cublasOperation_t_string(transa), mtk::ozimma::CULiP::get_cublasOperation_t_string(transb), m, n, k);
			mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		mtk::ozimma::gemm(
				get_global_ozimma_handle(),
				op_cublas2oz(transa),
				op_cublas2oz(transb),
				m, n, k,
				alpha,
				A, lda,
				B, ldb,
				beta,
				C, ldc,
				compute_mode
				);

		if (profiling_flag) {
			// Record end rimestamp
			mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::print_profile_result, (void*)&profile_result);
		}

		return CUBLAS_STATUS_SUCCESS;
	} else {
		cudaStream_t cuda_stream;
		cublasGetStream(handle, &cuda_stream);

		mtk::ozimma::CULiP::profile_result profile_result;
		const auto profiling_flag = mtk::ozimma::CULiP::is_profiling_enabled();

		cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
		*(void**)(&func_ptr) = ozIMMA_get_function_pointer(
				cublas_library_name.c_str(),
				__func__
				);

		if (profiling_flag) {
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%d-n%d-k%d", __func__, mtk::ozimma::CULiP::get_cublasOperation_t_string(transa), mtk::ozimma::CULiP::get_cublasOperation_t_string(transb), m, n, k);
			mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

		if (profiling_flag) {
			// Record end rimestamp
			mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::print_profile_result, (void*)&profile_result);
		}

		return res;
	}
#endif
}

CUBLASAPI cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A,
                            cudaDataType_t Atype, int lda, const void *B,
                            cudaDataType_t Btype, int ldb, const void *beta,
														void *C, cudaDataType_t Ctype, int ldc,
														cublasComputeType_t computeType,
														cublasGemmAlgo_t algo) {
#ifdef __CUDA_ARCH__
	return CUBLAS_STATUS_NOT_SUPPORTED;
#else
	if (Atype == CUDA_R_64F && Btype == CUDA_R_64F && Ctype == CUDA_R_64F) {
		return cublasDgemm_v2(
				handle,
				transa, transb,
				m, n, k,
				reinterpret_cast<const double*>(alpha),
				reinterpret_cast<const double*>(A), lda,
				reinterpret_cast<const double*>(B), ldb,
				reinterpret_cast<const double*>(beta),
				reinterpret_cast<double*>(C), ldc
				);
	}

	cudaStream_t cuda_stream;
	cublasGetStream(handle, &cuda_stream);

	mtk::ozimma::CULiP::profile_result profile_result;
	const auto profiling_flag = mtk::ozimma::CULiP::is_profiling_enabled();

	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = ozIMMA_get_function_pointer(
			cublas_library_name.c_str(),
			__func__
			);

	if (profiling_flag) {
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%d-n%d-k%d", __func__, mtk::ozimma::CULiP::get_cublasOperation_t_string(transa), mtk::ozimma::CULiP::get_cublasOperation_t_string(transb), m, n, k);
		mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
	}

	const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

	if (profiling_flag) {
		// Record end rimestamp
		mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		mtk::ozimma::CULiP::launch_function(cuda_stream, &mtk::ozimma::CULiP::print_profile_result, (void*)&profile_result);
	}

	return res;
#endif
}
} // extern "C"
