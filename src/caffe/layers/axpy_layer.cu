/*
 * Axpy Layer
 *
 * Created on: May 1, 2017
 * Author: hujie
 */

#include <device_launch_parameters.h>
#include "caffe/util/half.cuh"
#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/layers/axpy_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AxpyForward(const int count, const int spatial_dim,
    const Dtype* scale_data, const Dtype* x_data, const Dtype* y_data,
    Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, count) {
    out_data[index] = scale_data[index / spatial_dim] * x_data[index]
        + y_data[index];
  }
}

template <typename Ftype, typename Btype>
void AxpyLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const Ftype* scale_data = bottom[0]->gpu_data<Ftype>();
  const Ftype* x_data = bottom[1]->gpu_data<Ftype>();
  const Ftype* y_data = bottom[2]->gpu_data<Ftype>();
  Ftype* out_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[1]->count();
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  AxpyForward<Ftype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, bottom[1]->count(2), scale_data, x_data, y_data, out_data);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename Dtype>
__global__ void AxpyBackwardScale(const int outer_num, const int spatial_dim,
    const Dtype* x_data, const Dtype* top_diff, Dtype* scale_diff) {
  __shared__ char axpy_buffer[CAFFE_CUDA_NUM_THREADS * sizeof(Dtype)];
  Dtype* buffer = reinterpret_cast<Dtype*>(axpy_buffer);
  unsigned int tid = threadIdx.x;
  buffer[tid] = 0;
  __syncthreads();

  for (int j = tid; j < spatial_dim; j += blockDim.x) {
    int offset = blockIdx.x * spatial_dim + j;
    buffer[tid] += top_diff[offset] * x_data[offset];
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_diff[blockIdx.x] = buffer[0];
  }
}

template <typename Dtype>
__global__ void AxpyBackwardX(const int count, const int spatial_dim,
    const Dtype* scale_data, const Dtype* top_diff, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = scale_data[index / spatial_dim] * top_diff[index];
  }
}

template <typename Ftype, typename Btype>
void AxpyLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  const int count = top[0]->count();
  const Btype* top_diff = top[0]->gpu_diff<Btype>();
  if (propagate_down[0]) {
    cudaStream_t stream = Caffe::thread_stream();
    int outer_num = bottom[1]->count(0, 2);
    // NOLINT_NEXT_LINE(whitespace/operators)
    AxpyBackwardScale<<<outer_num, CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
        outer_num, bottom[1]->count(2),
        bottom[1]->gpu_data<Btype>(), top_diff,
        bottom[0]->mutable_gpu_diff<Btype>());
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  if (propagate_down[1]) {
    cudaStream_t stream = Caffe::thread_stream();
    // NOLINT_NEXT_LINE(whitespace/operators)
    AxpyBackwardX<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
        count, top[0]->count(2),
        bottom[0]->gpu_data<Btype>(), top_diff,
        bottom[1]->mutable_gpu_diff<Btype>());
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  if (propagate_down[2]) {
    caffe_copy(count, top_diff, bottom[2]->mutable_gpu_diff<Btype>());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(AxpyLayer);

}  // namespace caffe
