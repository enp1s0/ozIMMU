#include "cublas_helper.hpp"
#include "handle.hpp"
#include "utils.hpp"

namespace {
template <class dst_t, class src_t> __device__ dst_t convert(const src_t s) {
  return static_cast<dst_t>(s);
}
template <>
__device__ cuDoubleComplex
convert<cuDoubleComplex, cuComplex>(const cuComplex s) {
  return make_cuDoubleComplex(s.x, s.y);
}
template <>
__device__ cuComplex
convert<cuComplex, cuDoubleComplex>(const cuDoubleComplex s) {
  return make_cuComplex(s.x, s.y);
}

template <class src_t, class dst_t>
__global__ void
convert_dtype_kernel(dst_t *const dst_ptr, const std::size_t ldd,
                     const std::size_t dst_stride, const src_t *const src_ptr,
                     const std::size_t lds, const std::size_t src_stride,
                     std::uint32_t m, const std::uint32_t n,
                     const std::size_t batch_size) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= m * n * batch_size) {
    return;
  }

  const auto batch_id = tid / (m * n);
  const auto local_tid = tid % (m * n);

  const auto im = local_tid % m;
  const auto in = local_tid / m;

  const auto src_index = im + in * lds + src_stride * batch_id;
  const auto dst_index = im + in * ldd + dst_stride * batch_id;

  dst_ptr[dst_index] = convert<dst_t>(src_ptr[src_index]);
}

template <class src_t, class dst_t>
void convert_dtype(dst_t *const dst_ptr, const std::size_t ldd,
                   const std::size_t dst_stride, const src_t *const src_ptr,
                   const std::size_t lds, const std::size_t src_stride,
                   const std::uint32_t m, const std::uint32_t n,
                   const std::size_t batch_size, const cublasOperation_t op,
                   cudaStream_t cuda_stream) {
  constexpr std::uint32_t block_size = 256;
  const auto grid_size = (m * n * batch_size + block_size - 1) / block_size;

  auto mo = op == CUBLAS_OP_N ? m : n;
  auto no = op == CUBLAS_OP_N ? n : m;

  convert_dtype_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
      dst_ptr, ldd, dst_stride, src_ptr, lds, src_stride, mo, no, batch_size);
}

template <class src_t, class dst_t>
void convert_dtype(dst_t *const dst_ptr, const std::size_t ldd,
                   const src_t *const src_ptr, const std::size_t lds,
                   const std::uint32_t m, const std::uint32_t n,
                   const cublasOperation_t op, cudaStream_t cuda_stream) {
  convert_dtype(dst_ptr, ldd, 0, src_ptr, lds, 0, m, n, 1, op, cuda_stream);
}

template <class f64_t> struct f32_type {
  using type = void;
};
template <> struct f32_type<double> {
  using type = float;
};
template <> struct f32_type<cuDoubleComplex> {
  using type = cuComplex;
};

bool is_zero(const double v) { return v == 0; }
bool is_zero(const cuDoubleComplex v) { return v.x == 0 && v.y == 0; }
} // unnamed namespace

template <class T>
cublasStatus_t mtk::ozimmu::dgemm_f32(
    mtk::ozimmu::handle_t handle, cublasOperation_t op_A,
    cublasOperation_t op_B, const std::size_t m, const std::size_t n,
    const std::size_t k, const T alpha, const T *const mat_A_ptr,
    const std::size_t lda, const T *const mat_B_ptr, const std::size_t ldb,
    const T beta, T *const mat_C_ptr, const std::size_t ldc) {

  static_assert(std::is_same_v<T, double> ||
                std::is_same_v<T, cuDoubleComplex>);

  // Allocate mem space for the converted FP32 matrices A B and C
  const auto mem_space_size_for_f32 = (m * n + n * k + k * m) * sizeof(float);
  mtk::ozimmu::reallocate_working_memory(handle, mem_space_size_for_f32);

  auto A_f32_ptr = reinterpret_cast<typename f32_type<T>::type *>(
      handle->working_memory_ptr);
  auto B_f32_ptr = A_f32_ptr + m * k;
  auto C_f32_ptr = B_f32_ptr + n * k;

  const auto ld_f32_A = (op_A == CUBLAS_OP_N ? m : k);
  const auto ld_f32_B = (op_B == CUBLAS_OP_N ? k : n);

  convert_dtype(A_f32_ptr, ld_f32_A, mat_A_ptr, lda, m, k, op_A,
                handle->cuda_stream);
  convert_dtype(B_f32_ptr, ld_f32_B, mat_B_ptr, ldb, k, n, op_B,
                handle->cuda_stream);
  if (!is_zero(beta)) {
    convert_dtype(C_f32_ptr, m, mat_C_ptr, ldc, m, n, CUBLAS_OP_N,
                  handle->cuda_stream);
  }

  cublasStatus_t sgemm_status;
  if constexpr (std::is_same_v<double, T>) {
    const auto alpha_f32 = static_cast<float>(alpha);
    const auto beta_f32 = static_cast<float>(beta);
    sgemm_status = cublasSgemm(handle->cublas_handle, op_A, op_B, m, n, k,
                               &alpha_f32, A_f32_ptr, ld_f32_A, B_f32_ptr,
                               ld_f32_B, &beta_f32, C_f32_ptr, m);
  } else {
    const auto alpha_f32 = make_cuComplex(alpha.x, alpha.y);
    const auto beta_f32 = make_cuComplex(beta.x, beta.y);
    sgemm_status = cublasCgemm(handle->cublas_handle, op_A, op_B, m, n, k,
                               &alpha_f32, A_f32_ptr, ld_f32_A, B_f32_ptr,
                               ld_f32_B, &beta_f32, C_f32_ptr, m);
  }

  convert_dtype(mat_C_ptr, ldc, C_f32_ptr, m, m, n, CUBLAS_OP_N,
                handle->cuda_stream);

  return sgemm_status;
}

#define DGEMM_F32_INSTANCE(T)                                                  \
  template cublasStatus_t mtk::ozimmu::dgemm_f32(                              \
      mtk::ozimmu::handle_t handle, cublasOperation_t op_A,                    \
      cublasOperation_t op_B, const std::size_t m, const std::size_t n,        \
      const std::size_t k, const T alpha, const T *const mat_A_ptr,            \
      const std::size_t lda, const T *const mat_B_ptr, const std::size_t ldb,  \
      const T beta, T *const mat_C_ptr, const std::size_t ldc)

DGEMM_F32_INSTANCE(double);
DGEMM_F32_INSTANCE(cuDoubleComplex);

template <class T>
cublasStatus_t mtk::ozimmu::dgemm_f32_batched(
    mtk::ozimmu::handle_t handle, cublasOperation_t op_A,
    cublasOperation_t op_B, const std::size_t m, const std::size_t n,
    const std::size_t k, const T alpha, const T *const mat_A_ptr,
    const std::size_t lda, const std::size_t stride_a, const T *const mat_B_ptr,
    const std::size_t ldb, const std::size_t stride_b, const T beta,
    T *const mat_C_ptr, const std::size_t ldc, const std::size_t stride_c,
    const std::size_t batch_size) {

  static_assert(std::is_same_v<T, double> ||
                std::is_same_v<T, cuDoubleComplex>);

  // Allocate mem space for the converted FP32 matrices A B and C
  const auto mem_space_size_for_f32 =
      (m * n + n * k + k * m) * sizeof(float) * batch_size;
  mtk::ozimmu::reallocate_working_memory(handle, mem_space_size_for_f32);

  auto A_f32_ptr = reinterpret_cast<typename f32_type<T>::type *>(
      handle->working_memory_ptr);
  auto B_f32_ptr = A_f32_ptr + m * k * batch_size;
  auto C_f32_ptr = B_f32_ptr + n * k * batch_size;

  const auto ld_f32_A = (op_A == CUBLAS_OP_N ? m : k);
  const auto ld_f32_B = (op_B == CUBLAS_OP_N ? k : n);

  convert_dtype(A_f32_ptr, ld_f32_A, m * k, mat_A_ptr, lda, stride_a, m, k,
                batch_size, op_A, handle->cuda_stream);
  convert_dtype(B_f32_ptr, ld_f32_B, n * k, mat_B_ptr, ldb, stride_b, k, n,
                batch_size, op_B, handle->cuda_stream);
  if (!is_zero(beta)) {
    convert_dtype(C_f32_ptr, m, m * n, mat_C_ptr, ldc, stride_c, m, n,
                  batch_size, CUBLAS_OP_N, handle->cuda_stream);
  }

  cublasStatus_t sgemm_status;
  if constexpr (std::is_same_v<double, T>) {
    const auto alpha_f32 = static_cast<float>(alpha);
    const auto beta_f32 = static_cast<float>(beta);
    sgemm_status = cublasSgemmStridedBatched(
        handle->cublas_handle, op_A, op_B, m, n, k, &alpha_f32, A_f32_ptr,
        ld_f32_A, m * k, B_f32_ptr, ld_f32_B, n * k, &beta_f32, C_f32_ptr, m,
        m * n, batch_size);
  } else {
    const auto alpha_f32 = make_cuComplex(alpha.x, alpha.y);
    const auto beta_f32 = make_cuComplex(beta.x, beta.y);
    sgemm_status = cublasCgemmStridedBatched(
        handle->cublas_handle, op_A, op_B, m, n, k, &alpha_f32, A_f32_ptr,
        ld_f32_A, m * k, B_f32_ptr, ld_f32_B, n * k, &beta_f32, C_f32_ptr, m,
        m * n, batch_size);
  }

  convert_dtype(mat_C_ptr, ldc, stride_c, C_f32_ptr, m, m * n, m, n, batch_size,
                CUBLAS_OP_N, handle->cuda_stream);

  return sgemm_status;
}

#define DGEMM_F32_BATCHED_INSTANCE(T)                                          \
  template cublasStatus_t mtk::ozimmu::dgemm_f32_batched<T>(                   \
      mtk::ozimmu::handle_t handle, cublasOperation_t op_A,                    \
      cublasOperation_t op_B, const std::size_t m, const std::size_t n,        \
      const std::size_t k, const T alpha, const T *const mat_A_ptr,            \
      const std::size_t lda, const std::size_t stride_a,                       \
      const T *const mat_B_ptr, const std::size_t ldb,                         \
      const std::size_t stride_b, const T beta, T *const mat_C_ptr,            \
      const std::size_t ldc, const std::size_t stride_c,                       \
      const std::size_t batch_size);
DGEMM_F32_BATCHED_INSTANCE(double);
DGEMM_F32_BATCHED_INSTANCE(cuDoubleComplex);
