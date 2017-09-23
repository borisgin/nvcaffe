#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
}

template<typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // Add remaining top blob diffs.
  for (int i = 1; i < top.size(); ++i) {
    caffe_gpu_incr(count_, top[i]->gpu_diff<Btype>(), bottom[0]->mutable_gpu_diff<Btype>());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(SplitLayer);

}  // namespace caffe
