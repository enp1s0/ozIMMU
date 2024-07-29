#pragma once
#include <ctime>
#include <cutf/cublas.hpp>

namespace mtk {
namespace ozimmu {
namespace CULiP {
// Profile result
struct profile_result {
  // name
  enum { function_name_length = 128 };
  char function_name[function_name_length] = {0};

  // tm
  struct timespec start_timestamp;
  struct timespec end_timestamp;
};

void record_timestamp(void *tm_timestamp);
void print_profile_result(void *profile_result_ptr);
void launch_function(cudaStream_t cuda_stream, void (*fn)(void *),
                     void *const arg);

bool is_profiling_enabled();

const char *
get_cublasComputeType_t_string(const cublasComputeType_t compute_type);
const char *get_cublasOperation_t_string(const cublasOperation_t op);
} // namespace CULiP
} // namespace ozimmu
} // namespace mtk
