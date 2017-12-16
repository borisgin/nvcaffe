#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Note: this function runs in one thread per object and one object per one Solver thread
  shared_ptr<Batch> batch = this->batch_transformer_->processed_pop();
  top[0]->Swap(*batch->data_);
  if (this->output_labels_) {
    top[1]->Swap(*batch->label_);
  }
  this->batch_transformer_->processed_push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(BasePrefetchingDataLayer);

}  // namespace caffe
