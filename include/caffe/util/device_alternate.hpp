#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifndef NO_NVML
  #include <nvml.h>
#endif
#include <sched.h>

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_CHECK_ARG(condition, arg) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error) << \
        " (" << arg << ")"; \
  } while (0)

#define CUDA_CHECK_ARG2(condition, arg1, arg2) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error) << \
        " (" << arg1 << ") (" << arg2 << ")"; \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

#define CURAND_CHECK_ARG(condition, arg) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status) << \
        " (" << arg << ")"; \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;
const int CAFFE_CUDA_NUM_THREADS_HALF = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
inline int CAFFE_GET_BLOCKS_HALF(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS_HALF - 1) /
      CAFFE_CUDA_NUM_THREADS_HALF;
}


#ifndef NO_NVML
namespace nvml {

// We might move this to Caffe TLS but we have to make sure that
// this one gets initialized immediately after thread start.
// Also, it's better to run this on current device (note that Caffe ctr
// might be executed somewhere else). So, let's keep it risk free.
struct NVMLInit {
  NVMLInit();
  ~NVMLInit();
  static std::mutex m_;
};

void setCpuAffinity(int device);

}
#endif  // NO_NVML

}  // namespace caffe

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
