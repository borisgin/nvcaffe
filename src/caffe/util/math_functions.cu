#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/util/half.cuh"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/type.hpp"

namespace caffe {

template<>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(0), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
}

template<>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(0), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
}

template<>
void caffe_gpu_gemm<float16>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float16 alpha, const float16* A, const float16* B, const float16 beta,
    float16* C) {
  cublasHandle_t handle = Caffe::cublas_handle(0);
  // Note that cublas follows fortran order.
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  if (Caffe::device_capability(Caffe::current_device()) >= 503) {
#if CUDA_VERSION >= 9000
    cublasMath_t math_mode;
    CUBLAS_CHECK(cublasGetMathMode(handle, &math_mode));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    const float alpha_fp32 = static_cast<float>(alpha);
    const float beta_fp32 = static_cast<float>(beta);
    CUBLAS_CHECK(cublasGemmEx(handle, cuTransB, cuTransA,
        N, M, K, &alpha_fp32, B->gethp<half>(), CUDA_R_16F, ldb,
        A->gethp<half>(), CUDA_R_16F, lda, &beta_fp32, C->gethp<half>(),
        CUDA_R_16F, N, CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CUBLAS_CHECK(cublasSetMathMode(handle, math_mode));
#else
    CUBLAS_CHECK(cublasHgemm(handle, cuTransB, cuTransA,
    N, M, K, alpha.gethp<half>(), B->gethp<half>(), ldb,
    A->gethp<half>(), lda, beta.gethp<half>(), C->gethp<half>(), N));
#endif
  } else {
    float alpha_fp32 = static_cast<float>(alpha);
    float beta_fp32 = static_cast<float>(beta);
    CUBLAS_CHECK(cublasSgemmEx(handle, cuTransB, cuTransA,
        N, M, K, &alpha_fp32, B->gethp<half>(), CAFFE_DATA_HALF, ldb,
        A->gethp<half>(), CAFFE_DATA_HALF, lda, &beta_fp32, C->gethp<half>(),
        CAFFE_DATA_HALF, N));
  }
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
}

template<>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(0), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
}

template<>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(0), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
}

template<>
void caffe_gpu_gemv<float16>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float16 alpha, const float16* A, const float16* x,
    const float16 beta, float16* y) {
  cublasHandle_t handle = Caffe::cublas_handle(0);
  cublasOperation_t cuTransA = TransA == CblasNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = cuTransA == CUBLAS_OP_N ? N : M;
  int k = cuTransA == CUBLAS_OP_N ? M : N;
  int LDA = cuTransA == CUBLAS_OP_N ? m : k;
  int LDC = m;

  if (Caffe::device_capability(Caffe::current_device()) >= 503) {
#if CUDA_VERSION >= 9000
    cublasMath_t math_mode;
    CUBLAS_CHECK(cublasGetMathMode(handle, &math_mode));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    const float alpha_fp32 = static_cast<float>(alpha);
    const float beta_fp32 = static_cast<float>(beta);
    CUBLAS_CHECK(cublasGemmEx(handle, cuTransA, CUBLAS_OP_N,
        m, 1, k, &alpha_fp32, A->gethp<half>(), CUDA_R_16F, LDA,
        x->gethp<half>(), CUDA_R_16F, k, &beta_fp32, y->gethp<half>(),
        CUDA_R_16F, LDC, CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CUBLAS_CHECK(cublasSetMathMode(handle, math_mode));
#else
    CUBLAS_CHECK(cublasHgemm(handle, cuTransA, CUBLAS_OP_N,
        m, 1, k, alpha.gethp<half>(), A->gethp<half>(), LDA,
        x->gethp<half>(), k, beta.gethp<half>(),
        y->gethp<half>(), LDC));
#endif
  } else {
    float alpha_fp32 = static_cast<float>(alpha);
    float beta_fp32 = static_cast<float>(beta);
    CUBLAS_CHECK(cublasSgemmEx(Caffe::cublas_handle(0), cuTransA, CUBLAS_OP_N,
        m, 1, k, &alpha_fp32, A, CAFFE_DATA_HALF, LDA,
        x, CAFFE_DATA_HALF, k, &beta_fp32,
        y, CAFFE_DATA_HALF, LDC));
  }
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(0)));
}

template<>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y, void* handle) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y, void* handle) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype, typename Mtype>
__global__
void axpy_kernel(const int N, const Mtype alpha, const Dtype* x, Dtype* y) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < N;
       idx += blockDim.x * gridDim.x) {
    y[idx] = alpha * (Mtype) x[idx] + (Mtype) y[idx];
  }
}

template<>
__global__
void axpy_kernel<half, half>(const int N, const half alpha, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530
  CUDA_KERNEL_LOOP(idx, N) {
    y[idx] = __hfma(alpha, x[idx], y[idx]);
  }
#else
  CUDA_KERNEL_LOOP(idx, N) {
    y[idx] = float2half_clip(__half2float(y[idx]) + __half2float(alpha) * __half2float(x[idx]));
  }
#endif
}

template<>
void caffe_gpu_axpy<float16>(const int N, const float16 alpha, const float16* x, float16* y,
    void* handle) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  half ha;
  ha.setx(alpha.getx());
  // NOLINT_NEXT_LINE(whitespace/operators)
  axpy_kernel<<<CAFFE_GET_BLOCKS_HALF(N), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (N, ha, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y, int group) {
  if (X != Y) {
    cudaStream_t stream = Caffe::thread_stream(group);
    CUDA_CHECK(cudaMemcpyAsync(Y, X, N, cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

__global__
void scale_in_place_kernel(const int n, const half alpha, half* x) {
  CUDA_KERNEL_LOOP(idx, n) {
    x[idx] = hmul(alpha, x[idx]);
  }
}

template<>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X, cublasHandle_t cublas_handle) {
  if (alpha == 1.F) { return; }
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  CUBLAS_CHECK(cublasSscal(cublas_handle, N, &alpha, X, 1));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
    cublasHandle_t cublas_handle) {
  if (alpha == 1.0) { return; }
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  CUBLAS_CHECK(cublasDscal(cublas_handle, N, &alpha, X, 1));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_scal<float16>(const int n, const float16 alpha, float16* x,
    cublasHandle_t cublas_handle) {
  if (alpha.getx() == 0x3c00U) { return; }
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  half ha;
  ha.setx(alpha.getx());
  // use cublasHscal when it will become available
  // NOLINT_NEXT_LINE(whitespace/operators)
  scale_in_place_kernel <<<CAFFE_GET_BLOCKS_HALF(n), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (n, ha, reinterpret_cast<half*>(x));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X) {
  caffe_gpu_scal(N, alpha, X, Caffe::cublas_handle(0));
}

template<>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X) {
  caffe_gpu_scal(N, alpha, X, Caffe::cublas_handle(0));
}

template<>
void caffe_gpu_scal<float16>(const int N, const float16 alpha, float16* X) {
  caffe_gpu_scal(N, alpha, X, Caffe::cublas_handle(0));
}

template<>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template<>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template<typename Dtype, typename Mtype>
__global__
void axpby_kernel(const int N, const Mtype alpha, const Dtype* X, const Mtype beta, Dtype* Y) {
  CUDA_KERNEL_LOOP(idx, N) {
    Y[idx] = alpha * X[idx] + beta * Y[idx];
  }
}

template<>
void caffe_gpu_axpby<float16>(const int N, const float16 alpha,
    const float16* X, const float16 beta, float16* Y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  axpby_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, alpha, X, beta, Y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_dot<float, float>(const int n, const float* x, const float* y, float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(0), n, x, 1, y, 1, out));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template<>
void caffe_gpu_dot<double, double>(const int n, const double* x, const double* y, double* out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(0), n, x, 1, y, 1, out));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template<>
void caffe_gpu_dot<double, float>(const int n, const double* x, const double* y, float* outf) {
  double out = 0.;
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(0), n, x, 1, y, 1, &out));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  *outf = static_cast<float>(out);
}

template<typename Dtype, typename Mtype>
__global__
void gpu_dot_kernel(const int N, const Dtype* x, const Dtype* y, Mtype* out) {
  __shared__
  Mtype cache[CAFFE_CUDA_NUM_THREADS];
  const int tidx = threadIdx.x;
  cache[tidx] = 0.;
  __syncthreads();
  for (int i = tidx; i < N; i += blockDim.x) {
    cache[tidx] += static_cast<Mtype>(x[i]) * static_cast<Mtype>(y[i]);
  }
  __syncthreads();
  for (int s = CAFFE_CUDA_NUM_THREADS / 2; s > 0; s >>= 1) {
    if (tidx < s) cache[tidx] += cache[tidx + s];
    __syncthreads();
  }
  if (tidx == 0) *out = cache[tidx];
}

template<>
void
caffe_gpu_dot<float16, float16>(const int n, const float16* x, const float16* y, float16* out) {
  float fres;
  GPUMemory::Workspace ws(sizeof(float), Caffe::current_device());
  float* res = reinterpret_cast<float*>(ws.data());
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  gpu_dot_kernel<<<1, CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y, res);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaMemcpyAsync(&fres, res, ws.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  *out = static_cast<float16>(fres);
}

template<>
void caffe_gpu_dot<float16, float>(const int n, const float16* x, const float16* y, float* out) {
  GPUMemory::Workspace ws(sizeof(float), Caffe::current_device());
  float* res = reinterpret_cast<float*>(ws.data());
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  gpu_dot_kernel<<<1, CAFFE_CUDA_NUM_THREADS, 0, stream>>>(n, x, y, res);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaMemcpyAsync(out, res, ws.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_asum<float, float>(const int n, const float* x, float* y, int group) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(group), n, x, 1, y));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(group)));
}

template<>
void caffe_gpu_asum<float, double>(const int n, const float* x, double* y, int group) {
  float yf;
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(group), n, x, 1, &yf));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(group)));
  *y = yf;
}
template<>
void caffe_gpu_asum<double, double>(const int n, const double* x, double* y, int group) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(group), n, x, 1, y));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(group)));
}
template<>
void caffe_gpu_asum<double, float>(const int n, const double* x, float* y, int group) {
  double yd;
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(group), n, x, 1, &yd));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream(group)));
  *y = yd;
}

template<>
void caffe_gpu_scale<double>(const int n, const double alpha, const double* x, double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(0), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(0), n, &alpha, y, 1));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

template<>
void caffe_gpu_scale<float>(const int n, const float alpha, const float* x, float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(0), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(0), n, &alpha, y, 1));
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
}

__global__
void scale_kernel(const int n, const half alpha, const half* x, half* y) {
  CUDA_KERNEL_LOOP(idx, n) {
    y[idx] = hmul(alpha, x[idx]);
  }
}

template<>
void caffe_gpu_scale<float16>(const int n, const float16 alpha, const float16* x, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  half ha;
  ha.setx(alpha.getx());
  // NOLINT_NEXT_LINE(whitespace/operators)
  scale_kernel <<<CAFFE_GET_BLOCKS_HALF(n), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (n, ha, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void set_kernel(const size_t n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template<typename Dtype>
void caffe_gpu_set(const size_t N, const Dtype alpha, Dtype* Y) {
  cudaStream_t stream = Caffe::thread_stream();
  if (alpha == 0) {
    CUDA_CHECK(cudaMemsetAsync(Y, 0, sizeof(Dtype) * N, stream));  // NOLINT(caffe/alt_fn)
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    set_kernel <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N, alpha, Y);
    CUDA_POST_KERNEL_CHECK;
  }
  CUDA_CHECK_ARG2(cudaStreamSynchronize(stream), stream, Caffe::current_device());
}

template void
caffe_gpu_set<int>(const size_t N, const int alpha, int* Y);
template void
caffe_gpu_set<float>(const size_t N, const float alpha, float* Y);
template void
caffe_gpu_set<double>(const size_t N, const double alpha, double* Y);
template void
caffe_gpu_set<float16>(const size_t N, const float16 alpha, float16* Y);

template<typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template<>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators
  add_scalar_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_add_scalar(const int N, const float16 alpha, float16* Y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, alpha, Y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template<>
__global__ void add_kernel<half>(const int n, const half* a, const half* b, half* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = hadd(a[index], b[index]);
  }
}

template<>
void caffe_gpu_add<float>(const int N, const float* a, const float* b, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_add<double>(const int N, const double* a, const double* b, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_add<float16>(const int N, const float16* a, const float16* b, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<<<CAFFE_GET_BLOCKS_HALF(N), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (N, reinterpret_cast<const half*>(a), reinterpret_cast<const half*>(b),
       reinterpret_cast<half*>(y));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void incr_kernel(const int n, const Dtype* a, Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    b[index] += a[index];
  }
}

template<>
__global__ void incr_kernel<half>(const int n, const half* a, half* b) {
  CUDA_KERNEL_LOOP(index, n) {
    b[index] = hadd(a[index], b[index]);
  }
}

template<>
void caffe_gpu_incr<float>(const int N, const float* a, float* b) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  incr_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_incr<double>(const int N, const double* a, double* b) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  incr_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_incr<float16>(const int N, const float16* a, float16* b) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  incr_kernel<<<CAFFE_GET_BLOCKS_HALF(N), CAFFE_CUDA_NUM_THREADS_HALF, 0, stream>>>
      (N, reinterpret_cast<const half*>(a), reinterpret_cast<half*>(b));
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template<>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b, float* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b, double* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_sub<float16>(const int N, const float16* a, const float16* b, float16* y) {
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, y);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace caffe
