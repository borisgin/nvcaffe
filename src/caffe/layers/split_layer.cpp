#include <vector>

#include "caffe/net.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << "SplitLayer " << this->type()
        << " does not allow in-place computation.";
    if (top[i]->count() != bottom[0]->count()) {
      top[i]->ReshapeLike(*bottom[0]);
    }
    top[i]->ShareData(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
  bottom[0]->ShareDiff(*top[0]);
}

template <typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
}

template <typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  // Add remaining top blob diffs.
  for (int i = 1; i < top.size(); ++i) {
    const Btype* top_diff = top[i]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    caffe_axpy<Btype>(count_, Btype(1.), top_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS_FB(SplitLayer);
REGISTER_LAYER_CLASS(Split);

}  // namespace caffe
