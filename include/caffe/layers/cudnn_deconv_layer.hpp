#ifndef CAFFE_CUDNN_DECONV_LAYER_HPP_
#define CAFFE_CUDNN_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of DeConvolutionLayer.
 *        Fallback to DeConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates deconvolution through forward kernels for filtering and
 * bias plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
*/
template<typename Ftype, typename Btype>
class CuDNNDeconvolutionLayer : public DeconvolutionLayer<Ftype, Btype> {
 public:
  explicit CuDNNDeconvolutionLayer(const LayerParameter& param)
    : DeconvolutionLayer<Ftype, Btype>(param),
      handles_setup_(false),
      forward_math_(tpmax<Ftype, float>()),
      backward_data_math_(tpmax<Btype, float>()),
      backward_filter_math_(tpmax<Btype, float>()) {}
  virtual ~CuDNNDeconvolutionLayer();
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

 protected:
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
                    const vector<Blob*>& bottom) override;

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;
  Type forward_math_, backward_data_math_, backward_filter_math_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_DECONV_LAYER_HPP_
