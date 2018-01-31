#include <algorithm>
#include <device_launch_parameters.h>

#include "caffe/common.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/type.hpp"

namespace caffe {

SHMEM(amax);
CAFFE_GPU_SHMEM(amax);

#define BLOCK_REDUCE_AMAX(TNUM) \
if (BlockSize >= (TNUM) * 2) { \
  if (tid < (TNUM)) { \
    tmax_replace(st, sdata[tid + (TNUM)]); \
  } \
  __syncthreads(); \
}

#if CUDA_VERSION >= 9000
#define REDUCE_AMAX(TNUM) \
if (tid + (TNUM) < thread_count) { \
  tmax_replace(st, sdata[tid + (TNUM)]); \
  __syncwarp(); \
}
#else
#define REDUCE_AMAX(TNUM) \
if (tid + (TNUM) < thread_count) { \
  tmax_replace(st, sdata[tid + (TNUM)]); \
  __syncthreads(); \
}
#endif

///////////////////////////////////// AMAX REDUCTION ///////////////////////////////////

template<unsigned int BlockSize, typename T>
__device__ void amax_reduce_block(volatile T *sdata, T my_max, unsigned int tid) {
  const int thread_count = blockDim.x * blockDim.y * blockDim.z;
  volatile T* st = sdata + tid;
  *st = my_max;
  __syncthreads();
  // do reduction in shared mem
  BLOCK_REDUCE_AMAX(256)
  BLOCK_REDUCE_AMAX(128)
  BLOCK_REDUCE_AMAX(64)
  if (tid < 32) {
    REDUCE_AMAX(32)
    REDUCE_AMAX(16)
    REDUCE_AMAX(8)
    REDUCE_AMAX(4)
    REDUCE_AMAX(2)
    REDUCE_AMAX(1)
  }
}

// Global variable used by amax_reduce_kernel to count how many blocks have finished
__device__ unsigned int amax_blocks_count[REDUCTION_GROUPS_MAX];

void set_amax_blocks_count(unsigned int cnt, int group, cudaStream_t stream) {
  CUDA_CHECK_ARG(cudaMemcpyToSymbolAsync(amax_blocks_count, &cnt, sizeof(unsigned int),
      group * sizeof(unsigned int), cudaMemcpyHostToDevice, stream), Caffe::current_device());
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__device__ void amax_reduce_blocks(const T *in, TR *out, unsigned int n) {
  struct __dyn_shmem_amax__<TR> amax_shmem;
  // first level of reduction:
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
  unsigned int gridSize = BlockSize * 2 * gridDim.x;
  TR my_max = tzero<TR>();
  // We reduce multiple elements per thread. The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  while (i < n) {
    if (IsPow2 || i + BlockSize < n) {
      my_max = tmax<T, TR>(tabs(in[i]), tabs(in[i + BlockSize]));
    } else {
      tmax_replace(&my_max, tabs(in[i]));
    }
    i += gridSize;
  }
  // do reduction in shared mem
  amax_reduce_block<BlockSize>(amax_shmem.getPtr(), my_max, tid);
  // write result for this block to global mem
  if (tid == 0)
    out[blockIdx.x] = amax_shmem.getPtr()[0];
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__global__ void amax_reduce_kernel(unsigned int n, const T *in, TR *out, int group) {
  amax_reduce_blocks<BlockSize, IsPow2>(in, out, n);
  if (gridDim.x > 1) {
    const unsigned int tid = threadIdx.x;
    struct __dyn_shmem_amax__<TR> amax_reduce_shmem;
    __shared__ bool last_amax_block;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(amax_blocks_count + group, gridDim.x);
      last_amax_block = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // The last block max-es the results of all other blocks
    if (last_amax_block) {
      int i = tid;
      TR my_max = tzero<TR>();

      while (i < gridDim.x) {
        if (my_max < out[i]) {
          my_max = out[i];
        }
        i += BlockSize;
      }
      amax_reduce_block<BlockSize>(amax_reduce_shmem.getPtr(), my_max, tid);
      if (tid == 0) {
        out[0] = amax_reduce_shmem.getPtr()[0];
        // reset blocks count so that next run succeeds
        amax_blocks_count[group] = 0U;
      }
    }
  }
}

template <typename T, typename TR>
void gpu_amax_t(const int n, const T* x, TR* result, int group) {
  CHECK_LT(group, REDUCTION_GROUPS_MAX);
  cudaStream_t stream = Caffe::thread_stream(group);
  const bool po2 = is_pow2(n);
  // See kernel for details
  CHECK_LE(CAFFE_CUDA_NUM_THREADS_HALF, 512);
  CHECK_GE(CAFFE_CUDA_NUM_THREADS_HALF, 128);
  const int threadsPerCta = CAFFE_CUDA_NUM_THREADS_HALF;
  const int nbrCtas = CAFFE_GET_BLOCKS_HALF(n);
  const int reduction_size = (nbrCtas + 1) * sizeof(TR);
  GPUMemory::Workspace ws(reduction_size, Caffe::current_device());
  TR* dev_ptr_max = reinterpret_cast<TR*>(ws.data());
  set_amax_blocks_count(0U, group, stream);
  if (po2 && n > CAFFE_CUDA_NUM_THREADS_HALF) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    amax_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, true><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_max, group);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    amax_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, false><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_max, group);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaMemcpyAsync(result, dev_ptr_max, sizeof(TR), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
void caffe_gpu_amax(const int n, const T* x, float* result, int group) {
  gpu_amax_t(n, x, result, group);
}
template void caffe_gpu_amax<double>(const int n, const double* x, float* y, int group);
template void caffe_gpu_amax<float>(const int n, const float* x, float* y, int group);
template<>
void caffe_gpu_amax<float16>(const int n, const float16* x, float* y, int group) {
  // For odd counts we allocate extra element to speed up kernels.
  // We have to keep it clean.
  cudaStream_t stream = Caffe::thread_stream(group);
  if (n & 1) {
    clean_last_element(const_cast<float16*>(x) + n, stream);
  }
  const int n2 = even(n) / 2;
  gpu_amax_t(n2, reinterpret_cast<const half2*>(x), y, group);
#ifdef DEBUG
  CHECK(!isnan(*y));
  CHECK(!isinf(*y));
#endif
}

}  // namespace caffe
