#include <device_launch_parameters.h>

#include "caffe/common.hpp"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

///////////////////////////////////// SUMSQ REDUCTION ///////////////////////////////////

template<unsigned int BlockSize, typename TR>
__device__ void sumsq_reduce_block(volatile TR *sdata, TR my_sum, unsigned int tid) {
  volatile TR* st = sdata + tid;
  tassign(st, my_sum);
  __syncthreads();

  // do reduction in shared mem
  if (BlockSize >= 512) {
    if (tid < 256) {
      tsum_replace(st, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (BlockSize >= 256) {
    if (tid < 128) {
      tsum_replace(st, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (BlockSize >= 128) {
    if (tid < 64) {
      tsum_replace(st, sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    for (int i = 32; i > 0; i >>= 1) {
      tsum_replace(st, sdata[tid + i]);
    }
  }
}


// Global variable used by amax_reduce_kernel to count how many blocks have finished
__device__ unsigned int sumsq_blocks_count_f = 0;
__device__ unsigned int sumsq_blocks_count_d = 0;
__device__ unsigned int sumsq_blocks_count_h = 0;

template<typename T>
__device__ __inline__
unsigned int* sumsq_blocks_count_ptr();
template<>
__device__ __inline__
unsigned int* sumsq_blocks_count_ptr<float>() {
  return &sumsq_blocks_count_f;
}
template<>
__device__ __inline__
unsigned int* sumsq_blocks_count_ptr<double>() {
  return &sumsq_blocks_count_d;
}
template<>
__device__ __inline__
unsigned int* sumsq_blocks_count_ptr<half2>() {
  return &sumsq_blocks_count_h;
}

template<typename T>
cudaError_t set_sumsq_blocks_count(unsigned int cnt);
template<>
cudaError_t set_sumsq_blocks_count<float>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(sumsq_blocks_count_f, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}
template<>
cudaError_t set_sumsq_blocks_count<double>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(sumsq_blocks_count_d, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}
template<>
cudaError_t set_sumsq_blocks_count<half2>(unsigned int cnt) {
  return cudaMemcpyToSymbolAsync(sumsq_blocks_count_h, &cnt, sizeof(unsigned int), 0,
      cudaMemcpyHostToDevice, Caffe::thread_stream());
}

template<typename T>
__device__ __inline__
void reset_sumsq_blocks_count();
template<>
void reset_sumsq_blocks_count<float>() {
  sumsq_blocks_count_f = 0;
}
template<>
__device__ __inline__
void reset_sumsq_blocks_count<double>() {
  sumsq_blocks_count_d = 0;
}
template<>
__device__ __inline__
void reset_sumsq_blocks_count<half2>() {
  sumsq_blocks_count_h = 0;
}

template<unsigned int BlockSize, bool IsPow2, typename T, typename TR>
__device__ void sumsq_reduce_blocks(const T *in, TR *out, unsigned int n) {
  struct __dyn_shmem__<n_bytes<sizeof(TR)>> sumsq_blocks_shmem;
  TR* partial_sumsq = reinterpret_cast<TR*>(sumsq_blocks_shmem.getPtr());

  // first level of reduction:
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
  unsigned int gridSize = BlockSize * 2 * gridDim.x;
  T t1, t2;
  TR my_sum = tzero<TR>();
  // We reduce multiple elements per thread. The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  while (i < n) {
    t1 = in[i];
    if (IsPow2 || i + BlockSize < n) {
      t2 = in[i + BlockSize];
      tsum_replace(&my_sum, tsumsq<T, TR>(t1, t2));
    } else {
      tsumsq_replace(&my_sum, t1);
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
__global__ void sumsq_reduce_kernel(unsigned int n, const T *in, TR *out) {
  sumsq_reduce_blocks<BlockSize, IsPow2>(in, out, n);

  if (gridDim.x > 1) {
    const unsigned int tid = threadIdx.x;
    struct __dyn_shmem__<n_bytes<sizeof(TR)>> sumsq_reduce_shmem;
    TR* partial_sumsq = reinterpret_cast<TR*>(sumsq_reduce_shmem.getPtr());
    __shared__ bool last_sumsq_reduce_block;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(sumsq_blocks_count_ptr<T>(), gridDim.x);
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
        reset_sumsq_blocks_count<T>();
      }
    }
  }
}

template<typename T, typename TR>
void gpu_sumsq_t(const int n, const T* x, TR* sum, int group) {
  cudaStream_t stream = Caffe::thread_stream();
  const bool po2 = is_pow2(n);
  // See kernel for details
  CHECK_LE(CAFFE_CUDA_NUM_THREADS_HALF, 512);
  CHECK_GE(CAFFE_CUDA_NUM_THREADS_HALF, 128);
  const int threadsPerCta = CAFFE_CUDA_NUM_THREADS_HALF;
  const int nbrCtas = CAFFE_GET_BLOCKS_HALF(n);
  const int reduction_size_sum = (nbrCtas + 1) * sizeof(TR);
  TR* dev_ptr_sum = reinterpret_cast<TR*>(GPUMemory::thread_pinned_buffer(reduction_size_sum, group));
  if (po2 && n > CAFFE_CUDA_NUM_THREADS_HALF) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    sumsq_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, true><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_sum);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    sumsq_reduce_kernel<CAFFE_CUDA_NUM_THREADS_HALF, false><<<nbrCtas, threadsPerCta,
        threadsPerCta * sizeof(TR) + sizeof(bool), stream>>>
            ((unsigned int)n, x, dev_ptr_sum);
  }
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  *sum = dev_ptr_sum[0];
}

template<>
void caffe_gpu_sumsq<float16, float>(const int n, const float16* x, float* sum, int group) {
  // For odd counts we allocate extra element to speed up kernels.
  // We have to keep it clean.
  cudaStream_t stream = Caffe::thread_stream();
  if (n & 1) {
    clean_last_element(const_cast<float16*>(x) + n, stream);
  }
  const int n2 = even(n) / 2;
  static cudaError_t status = set_sumsq_blocks_count<half2>(0U);  // needed just 1 time
  CUDA_CHECK(status);
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
  static cudaError_t status = set_sumsq_blocks_count<Dtype>(0U);  // needed just 1 time
  CUDA_CHECK(status);
  gpu_sumsq_t(n, x, s, group);
}

template
void caffe_gpu_sumsq<float, float>(const int n, const float* x, float* s, int group);

template
void caffe_gpu_sumsq<double, float>(const int n, const double* x, float* s, int group);

}  // namespace caffe
