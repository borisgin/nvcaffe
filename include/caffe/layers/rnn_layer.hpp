#ifndef CAFFE_RNN_LAYER_HPP_
#define CAFFE_RNN_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Processes time-varying inputs using a simple recurrent neural network
 *        (RNN). Implemented as a network unrolling the RNN computation in time.
 *
 * Given time-varying inputs @f$ x_t @f$, computes hidden state @f$
 *     h_t := \tanh[ W_{hh} h_{t_1} + W_{xh} x_t + b_h ]
 * @f$, and outputs @f$
 *     o_t := \tanh[ W_{ho} h_t + b_o ]
 * @f$.
 */
template<typename Ftype, typename Btype>
class RNNLayer : public RecurrentLayer<Ftype, Btype> {
 public:
  explicit RNNLayer(const LayerParameter& param)
      : RecurrentLayer<Ftype, Btype>(param) {}

  virtual inline const char* type() const { return "RNN"; }

 protected:
  void FillUnrolledNet(NetParameter* net_param) const override;
  void RecurrentInputBlobNames(vector<string>* names) const override;
  void RecurrentOutputBlobNames(vector<string>* names) const override;
  void RecurrentInputShapes(vector<BlobShape>* shapes) const override;
  void OutputBlobNames(vector<string>* names) const override;
};

}  // namespace caffe

#endif  // CAFFE_RNN_LAYER_HPP_
