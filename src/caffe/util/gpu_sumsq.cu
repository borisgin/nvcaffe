#include <device_launch_parameters.h>

#include "caffe/common.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"
#include "caffe/type.hpp"

namespace caffe {

SHMEM(sq);
CAFFE_GPU_SHMEM(sq);

#define BLOCK_REDUCE_ASUM(TNUM) \
if (BlockSize >= (TNUM) * 2) { \
  if (tid < (TNUM)) { \
    tsum_replace(st, sdata[tid + (TNUM)]); \
  } \
  __syncthreads(); \
}

#if CUDA_VERSION >= 9000
#define REDUCE_ASUM(TNUM) \
if (tid + (TNUM) < thread_count) { \
  tsum_replace(st, sdata[tid + (TNUM)]); \
  __syncwarp(); \
}
#else
#define REDUCE_ASUM(TNUM) \
if (tid + (TNUM) < thread_count) { \
  tsum_replace(st, sdata[tid + (TNUM)]); \
  __syncthreads(); \
}
#endif

///////////////////////////////////// SUMSQ REDUCTION ///////////////////////////////////

template<unsigned int BlockSize, typename TR>
__device__ void sumsq_reduce_block(volatile TR *sdata, TR my_sum, unsigned int tid) {
  const int thread_count = blockDim.x * blockDim.y * blockDim.z;
  volatile TR* st = sdata + tid;
  *st = my_sum;
  __syncthreads();
  // do reduction in shared mem
  BLOCK_REDUCE_ASUM(256)
  BLOCK_REDUCE_ASUM(128)
  BLOCK_REDUCE_ASUM(64)
  if (tid < 32) {
    REDUCE_ASUM(32)
    REDUCE_ASUM(16)
    REDUCE_ASUM(8)
    REDUCE_ASUM(4)
    REDUCE_ASUM(2)
    REDUCE_ASUM(1)
  }
}

// Global variable used by sumsq_reduce_kernel to count how many blocks have finished
__device__ unsigned int sumsq_blocks_count[REDUCTION_GROUPS_MAX];

void set_sumsq_blocks_count(unsigned int cnt, int group, cudaStream_t stream) {
  CUDA_CHECK_ARG(cudaMemcpyToSymbolAsync(sumsq_blocks_count, &cnt, sizeof(unsigned int),
      group * sizeof(unsigned int), cudaMemcpyHostToDevice, stream), Caffe::current_device());
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__device__ void sumsq_reduce_blocks(const T *in, TR *out, unsigned int n) {
  struct __dyn_shmem_sq__<n_bytes<sizeof(TR)>> sumsq_blocks_shmem;
  TR* partial_sumsq = reinterpret_cast<TR*>(sumsq_blocks_shmem.getPtr());

  // first level of reduction:
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
  unsigned int gridSize = BlockSize * 2 * gridDim.x;
  TR my_sum = tzero<TR>();
  // We reduce multiple elements per thread. The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  while (i < n) {
    if (IsPow2 || i + BlockSize < n) {
      tsum_replace(&my_sum, tsumsq<T, TR>(in[i], in[i + BlockSize]));
    } else {
      tsumsq_replace(&my_sum, in[i]);
    }
    i += gridSize;
  }

  // do reduction in shared mem
  sumsq_reduce_block<BlockSize>(partial_sumsq, my_sum, tid);
  // write result for this block to global mem
  if (tid == 0) {
    out[blockIdx.x] = partial_sumsq[0];
  }
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__global__ void sumsq_reduce_kernel(unsigned int n, const T *in, TR *out, int group) {
  sumsq_reduce_blocks<BlockSize, IsPow2>(in, out, n);
  if (gridDim.x > 1) {
    const unsigned int tid = threadIdx.x;
    struct __dyn_shmem_sq__<n_bytes<sizeof(TR)>> sumsq_reduce_shmem;
    TR* partial_sumsq = reinterpret_cast<TR*>(sumsq_reduce_shmem.getPtr());
    __shared__ bool last_sumsq_reduce_block;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(sumsq_blocks_count + group, gridDim.x);
      last_sumsq_reduce_block = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // The last block sums the results of all other blocks
    if (last_sumsq_reduce_block) {
      int i = tid;
      TR my_sum = tzero<TR>();

      while (i < gridDim.x) {
        tsum_replace(&my_sum, out[i]);
        i += BlockSize;
      }
      sumsq_reduce_block<BlockSize>(partial_sumsq, my_sum, tid);
      if (tid == 0) {
        out[0] = partial_sumsq[0];
        // reset blocks count so that next run succeeds
        sumsq_blocks_count[group] = 0U;
      }
    }
  }
}

template<typename T, typename TR>
void gpu_sumsq_t(const int n, const T* x, TR* sum, int group) {
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
  TR* dev_ptr_sq = reinterpret_cast<TR*>(ws.data());
  set_sumsq_blocks_count(0U, group, stream);
  if (po2 && n > CAFFE_CUDA_NUM_THREADS_HALF) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    sumsq_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, true><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_sq, group);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    sumsq_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, false><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_sq, group);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaMemcpyAsync(sum, dev_ptr_sq, sizeof(TR), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void caffe_gpu_sumsq<float16, float>(const int n, const float16* x, float* sum, int group) {
  // For odd counts we allocate extra element to speed up kernels.
  // We have to keep it clean.
  cudaStream_t stream = Caffe::thread_stream(group);
  if (n & 1) {
    clean_last_element(const_cast<float16*>(x) + n, stream);
  }
  const int n2 = even(n) / 2;
  gpu_sumsq_t(n2, reinterpret_cast<const half2*>(x), sum, group);
}
template<>
void caffe_gpu_sumsq<float16, double>(const int n, const float16* x, double* sum, int group) {
  float sf;
  caffe_gpu_sumsq(n, x, &sf, group);
  *sum = sf;
}
template<>
void caffe_gpu_sumsq<float16, float16>(const int n, const float16* x, float16* sum, int group) {
  float sf;
  caffe_gpu_sumsq(n, x, &sf, group);
  *sum = sf;
}

template <typename Dtype, typename Mtype>
void caffe_gpu_sumsq(const int n, const Dtype* x, Mtype* s, int group) {
  gpu_sumsq_t(n, x, s, group);
}

template
void caffe_gpu_sumsq<float, float>(const int n, const float* x, float* s, int group);
template
void caffe_gpu_sumsq<double, float>(const int n, const double* x, float* s, int group);

}  // namespace caffe
