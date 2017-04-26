#include <algorithm>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <cuda_fp16.h>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/gpu_math_functions.cuh"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0.;
      counts[index] = 0.;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
          min_dtype<Dtype>()));
      counts[index] = 1.;
    }
  }
}

template <>
__global__ void SoftmaxLossForwardGPU<__half>(const int nthreads,
    const __half* prob_data, const __half* label, __half* loss,
    const int num, const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_,
    __half* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(__half2float(label[n * spatial_dim + s]));
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index].x = 0U;
      counts[index].x = 0U;
    } else {
      loss[index] = float2half_clip(- log(max(__half2float(
          prob_data[n * dim + label_value * spatial_dim + s]),
          __half2float(min_dtype<__half>()))));
      counts[index].x = 1U;
    }
  }
}


template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Forward_gpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Ftype* prob_data = prob_.template gpu_data<Ftype>();
  const Ftype* label = bottom[1]->gpu_data<Ftype>();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Ftype* loss_data = bottom[0]->mutable_gpu_diff<Ftype>();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Ftype* counts = prob_.template mutable_gpu_diff<Ftype>();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  Ftype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Ftype valid_count = -1.F;
  // Only launch another CUDA kernel if we actually need the count of valid outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data<Ftype>()[0] = loss / get_normalizer(normalization_, valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0.F;
      }
      counts[index] = 0.F;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1.F;
      counts[index] = 1.F;
    }
  }
}

template <>
__global__ void SoftmaxLossBackwardGPU<__half>(const int nthreads, const __half* top,
    const __half* label, __half* bottom_diff, const int num, const int dim,
    const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, __half* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(__half2float(label[n * spatial_dim + s]));

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s].x = 0U;
      }
      counts[index].x = 0U;
    } else {
      const int idx = n * dim + label_value * spatial_dim + s;
      bottom_diff[idx] = float2half_clip(__half2float(bottom_diff[idx]) - 1.F);
      counts[index].x = 0x3c00U;  // 1.
    }
  }
}


template <typename Ftype, typename Btype>
void SoftmaxWithLossLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>(false);
    const Btype* prob_data = prob_.template gpu_data<Btype>();
    const Btype* top_data = top[0]->gpu_data<Btype>();
    caffe_gpu_memcpy(prob_.count() * sizeof(Btype), prob_data, bottom_diff);
    const Btype* label = bottom[1]->gpu_data<Btype>();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Btype* counts = prob_.template mutable_gpu_diff<Btype>(false);
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS, 0, Caffe::thread_stream()>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    Btype valid_count = -1.F;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Btype loss_weight = top[0]->cpu_diff<Btype>()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SoftmaxWithLossLayer);

}  // namespace caffe
