#ifndef CAFFE_SILENCE_LAYER_HPP_
#define CAFFE_SILENCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#define SILENCE_PERF 1

namespace caffe {

/**
 * @brief Ignores bottom blobs while producing no top blobs. (This is useful
 *        to suppress outputs during testing.)
 */
template <typename Ftype, typename Btype>
class SilenceLayer : public Layer<Ftype, Btype> {
 public:
  explicit SilenceLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) override;

  virtual inline const char* type() const { return "Silence"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const {
#ifdef SILENCE_PERF
    return 1;
#else
    return 0;
#endif
  }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {}
  // We can't define Forward_gpu here, since STUB_GPU will provide
  // its own definition.
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_SILENCE_LAYER_HPP_
