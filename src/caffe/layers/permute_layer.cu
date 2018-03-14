#include <algorithm>
#include <cfloat>
#include <vector>
#include <device_launch_parameters.h>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PermuteKernel(const int nthreads,
    Dtype* bottom_data, const bool forward, const int* permute_order,
    const int* old_steps, const int* new_steps, const int num_axes,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp_idx = index;
    int old_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      int order = permute_order[i];
      old_idx += (temp_idx / new_steps[i]) * old_steps[order];
      temp_idx %= new_steps[i];
    }
    if (forward) {
      top_data[index] = bottom_data[old_idx];
    } else {
      bottom_data[old_idx] = top_data[index];
    }
  }
}

template <>
__global__ void PermuteKernel<float16>(const int nthreads, float16* bottom_data,
                                       const bool forward, const int* permute_order,
                                       const int* old_steps, const int* new_steps,
                                       const int num_axes, float16* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp_idx = index;
    int old_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      int order = permute_order[i];
      old_idx += (temp_idx / new_steps[i]) * old_steps[order];
      temp_idx %= new_steps[i];
    }
    if (forward) {
      top_data[index] = bottom_data[old_idx];
    } else {
      bottom_data[old_idx] = top_data[index];
    }
  }
}


template <typename Ftype, typename Btype>
void PermuteLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  if (need_permute_) {
    Dtype* bottom_data = const_cast<Dtype*>(bottom[0]->gpu_data<Dtype>());
    Dtype* top_data = top[0]->mutable_gpu_data<Dtype>();
    int count = top[0]->count();
    const int* permute_order = permute_order_.gpu_data();
    const int* new_steps = new_steps_.gpu_data();
    const int* old_steps = old_steps_.gpu_data();
    bool foward = true;
    cudaStream_t stream = Caffe::thread_stream();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
        count, bottom_data, foward, permute_order, old_steps, new_steps,
        num_axes_, top_data);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    // If there is no need to permute, we share data to save memory.
    top[0]->ShareData(*bottom[0]);
  }
}

template <typename Ftype, typename Btype>
void PermuteLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (need_permute_) {
    Dtype* top_diff = top[0]->mutable_gpu_diff<Dtype>();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff<Dtype>();
    const int count = bottom[0]->count();
    const int* permute_order = permute_order_.gpu_data();
    const int* new_steps = new_steps_.gpu_data();
    const int* old_steps = old_steps_.gpu_data();
    bool foward = false;
    cudaStream_t stream = Caffe::thread_stream();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
        count, bottom_diff, foward, permute_order, old_steps, new_steps,
        num_axes_, top_diff);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    // If there is no need to permute, we share diff to save memory.
    bottom[0]->ShareDiff(*top[0]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(PermuteLayer);

}  // namespace caffe
