#include <cuda.h>
#include <dlfcn.h>
#include <iostream>
#include <string.h>
#include <unistd.h>

#include "culip.hpp"

const std::string CULIP_RESULT_PREFIX = "CULiP Result";
const std::string CULIP_EXP_STATS_PREFIX = "CULiP ExpStats";
const std::string CUMPSGEMM_ENABLE_CULIP_PROFILING_ENV_NAME =
    "OZIMMU_ENABLE_CULIP_PROFILING";

void mtk::ozimmu::CULiP::record_timestamp(void *tm_timestamp) {
  struct timespec *tm_ptr = (struct timespec *)tm_timestamp;
  clock_gettime(CLOCK_MONOTONIC, tm_ptr);
}

void mtk::ozimmu::CULiP::print_profile_result(void *profile_result_ptr) {
  const auto profile_result =
      *((mtk::ozimmu::CULiP::profile_result *)profile_result_ptr);

  const unsigned long elapsed_time_us =
      ((long)profile_result.end_timestamp.tv_sec -
       (long)profile_result.start_timestamp.tv_sec) *
          (long)1000000000 +
      ((long)profile_result.end_timestamp.tv_nsec -
       (long)profile_result.start_timestamp.tv_nsec);
  printf("[%s][%s] %luns\n", CULIP_RESULT_PREFIX.c_str(),
         profile_result.function_name, elapsed_time_us);
  fflush(stdout);
}

void mtk::ozimmu::CULiP::launch_function(cudaStream_t cuda_stream,
                                         void (*fn)(void *), void *const arg) {
  CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
  fn(arg);
  CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
}

bool mtk::ozimmu::CULiP::is_profiling_enabled() {
  const char *value = getenv(CUMPSGEMM_ENABLE_CULIP_PROFILING_ENV_NAME.c_str());
  if (value == NULL) {
    return false;
  }
  if (std::string(value) == "0") {
    return false;
  }
  return true;
}

#define CULiP_CUBLAS_COMPUTE_T_CASE_STRING(compute_type)                       \
  case compute_type:                                                           \
    return #compute_type
const char *mtk::ozimmu::CULiP::get_cublasComputeType_t_string(
    const cublasComputeType_t compute_type) {
  switch (compute_type) {
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_16F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_16F_PEDANTIC);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_FAST_16BF);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_FAST_16F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_FAST_TF32);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32F_PEDANTIC);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32I);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_32I_PEDANTIC);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_64F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUBLAS_COMPUTE_64F_PEDANTIC);
  default:
    break;
  }
  switch ((cudaDataType_t)compute_type) {
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_16BF);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_16F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_32F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_32I);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_64F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_8I);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_C_8U);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_16BF);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_16F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_32F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_32I);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_64F);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_8I);
    CULiP_CUBLAS_COMPUTE_T_CASE_STRING(CUDA_R_8U);
  default:
    return "Unknown";
  }
}

const char *
mtk::ozimmu::CULiP::get_cublasOperation_t_string(const cublasOperation_t op) {
  switch (op) {
  case CUBLAS_OP_N:
    return "N";
  case CUBLAS_OP_T:
    return "T";
  case CUBLAS_OP_C:
    return "C";
  default:
    return "Unknown";
  }
}
