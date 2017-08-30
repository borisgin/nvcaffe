#include <vector>

#include "caffe/net.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*bottom[0]);
    top[i]->ShareData(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
  bottom[0]->ShareDiff(*top[0]);
}

template <typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ReshapeLike(*bottom[0]);
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Ftype, typename Btype>
void SplitLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy<Btype>(count_, top[0]->cpu_diff<Btype>(),
        bottom[0]->mutable_cpu_diff<Btype>());
    return;
  }
  caffe_add<Btype>(count_, top[0]->cpu_diff<Btype>(), top[1]->cpu_diff<Btype>(),
            bottom[0]->mutable_cpu_diff<Btype>());
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const Btype* top_diff = top[i]->cpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_cpu_diff<Btype>();
    caffe_axpy<Btype>(count_, Btype(1.), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SplitLayer);
#endif

INSTANTIATE_CLASS_FB(SplitLayer);
REGISTER_LAYER_CLASS(Split);

}  // namespace caffe
