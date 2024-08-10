#include <cublas_v2.h>
#include <ozimmu/ozimmu.hpp>

namespace mtk::ozimmu {
template <class T>
cublasStatus_t
dgemm_f32(mtk::ozimmu::handle_t handle, cublasOperation_t op_A,
          cublasOperation_t op_B, const std::size_t m, const std::size_t n,
          const std::size_t k, const T alpha, const T *const mat_A,
          const std::size_t lda, const T *const mat_B, const std::size_t ldb,
          const T beta, T *const mat_C, const std::size_t ldc);

template <class T>
cublasStatus_t
dgemm_f32_batched(mtk::ozimmu::handle_t handle, cublasOperation_t op_A,
                  cublasOperation_t op_B, const std::size_t m,
                  const std::size_t n, const std::size_t k, const T alpha,
                  const T *const mat_A, const std::size_t lda,
                  const std::size_t stride_a, const T *const mat_B,
                  const std::size_t ldb, const std::size_t stride_b,
                  const T beta, T *const mat_C, const std::size_t ldc,
                  const std::size_t stride_c, const std::size_t batch_size);
} // namespace mtk::ozimmu
