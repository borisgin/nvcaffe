#include <vector>

#include "caffe/layers/silence_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SilenceLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      bottom[i]->set_diff(0.F);
    }
  }
}

template <typename Ftype, typename Btype>
void SilenceLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
#ifdef SILENCE_PERF
  top[0]->Reshape(1);
#endif
}

INSTANTIATE_CLASS_FB(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe
