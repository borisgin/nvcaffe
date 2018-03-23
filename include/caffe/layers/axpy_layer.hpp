/*
 * Axpy Layer
 *
 * Created on: May 1, 2017
 * Author: hujie
 */

#ifndef CAFFE_AXPY_LAYER_HPP_
#define CAFFE_AXPY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief For reduce memory and time both on training and testing, we combine
 *        channel-wise scale operation and element-wise addition operation 
 *        into a single layer called "axpy".
 *       
 */
template <typename Ftype, typename Btype>
class AxpyLayer: public Layer<Ftype, Btype> {
 public:
  explicit AxpyLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {}
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual inline const char* type() const { return "Axpy"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
/**
 * @param Formulation:
 *            F = a * X + Y
 *	  Shape info:
 *            a:  N x C          --> bottom[0]      
 *            X:  N x C x H x W  --> bottom[1]       
 *            Y:  N x C x H x W  --> bottom[2]     
 *            F:  N x C x H x W  --> top[0]
 */
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

  TBlob<Btype> spatial_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_AXPY_LAYER_HPP_
