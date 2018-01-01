#ifdef USE_CUDNN

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void CuDNNBatchNormLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {

  BatchNormLayer<Ftype, Btype>::LayerSetUp(bottom, top);

  cudnn::createTensor4dDesc<Ftype>(&fwd_bottom_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_top_desc_);
  cudnn::createTensor4dDesc<Ftype>(&fwd_scale_bias_mean_var_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_bottom_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_top_desc_);
  cudnn::createTensor4dDesc<Btype>(&bwd_scale_bias_mean_var_desc_);

#if CUDNN_VERSION_MIN(7, 0, 0)
  mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  mode_ = CUDNN_BATCHNORM_SPATIAL;      // only SPATIAL mode is supported
#endif
  this->eps_ = std::max(this->eps_, CUDNN_BN_MIN_EPSILON);

  int channels = bottom[0]->channels();
  const Type btype = blobs_type();
  const vector<int> shape { 1, channels, 1, 1 };
  if (!this->scale_bias_) { // stubs for cudnn
    scale_ones_ = Blob::create(btype, btype);
    scale_ones_->Reshape(shape);
    scale_ones_->set_data(1.F);
    bias_zeros_ = Blob::create(btype, btype);
    bias_zeros_->Reshape(shape);
    bias_zeros_->set_data(0.F);
  }
  save_mean_->Reshape(shape);
  save_inv_var_->Reshape(shape);
  handles_setup_ = true;

  if (bottom == top) {  // CUDNN_BN does support in-place
    private_top_ = Blob::create<Ftype>(top[0]->shape());
    private_bottom_ = Blob::create<Ftype>(bottom[0]->shape());
  }
}

template<typename Ftype, typename Btype>
void
CuDNNBatchNormLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  BatchNormLayer<Ftype, Btype>::Reshape(bottom, top);

  int N = bottom[0]->num();
  int C = bottom[0]->channels();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
  // set up main tensors
  cudnn::setTensor4dDesc<Ftype>(&fwd_bottom_desc_, N, C, H, W);
  cudnn::setTensor4dDesc<Ftype>(&fwd_top_desc_, N, C, H, W);
  cudnn::setTensor4dDesc<Btype>(&bwd_bottom_desc_, N, C, H, W);
  cudnn::setTensor4dDesc<Btype>(&bwd_top_desc_, N, C, H, W);
  // aux tensors for caching mean & invVar from fwd to bwd pass
  save_mean_->Reshape(1, C, 1, 1);
  save_inv_var_->Reshape(1, C, 1, 1);
  if (!this->scale_bias_) {
    int C_old = scale_ones_->channels();
    if (C_old != C) {
      scale_ones_->Reshape(1, C, 1, 1);
      bias_zeros_->Reshape(1, C, 1, 1);
      scale_ones_->set_data(1.F);
      bias_zeros_->set_data(0.F);
    }
  }
  CUDNN_CHECK(
      cudnnDeriveBNTensorDescriptor(fwd_scale_bias_mean_var_desc_, fwd_bottom_desc_, mode_));
  CUDNN_CHECK(
      cudnnDeriveBNTensorDescriptor(bwd_scale_bias_mean_var_desc_, bwd_bottom_desc_, mode_));

  if (top[0] == bottom[0]) {
    if (!private_top_) {
      private_top_ = Blob::create<Ftype>(top[0]->shape());
    } else {
      private_top_->ReshapeLike(top[0]);
    }
    if (!private_bottom_) {
      private_bottom_ = Blob::create<Ftype>(bottom[0]->shape());
    } else {
      private_bottom_->ReshapeLike(bottom[0]);
    }
  }
}

template<typename Ftype, typename Btype>
CuDNNBatchNormLayer<Ftype, Btype>::~CuDNNBatchNormLayer() {
  if (!handles_setup_) return;
  cudnnDestroyTensorDescriptor(fwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(bwd_bottom_desc_);
  cudnnDestroyTensorDescriptor(fwd_top_desc_);
  cudnnDestroyTensorDescriptor(bwd_top_desc_);
  cudnnDestroyTensorDescriptor(fwd_scale_bias_mean_var_desc_);
  cudnnDestroyTensorDescriptor(bwd_scale_bias_mean_var_desc_);
}

INSTANTIATE_CLASS_FB(CuDNNBatchNormLayer);

}  // namespace caffe

#endif
