#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  // Note: this function runs in one thread per object and one object per one Solver thread
  shared_ptr<Batch> batch =
      prefetches_full_[next_batch_queue_]->pop("Data layer prefetch queue empty");
  if (batch->data_packing() == this->transform_param_.forward_packing()
      && top[0]->shape() == batch->data_->shape()) {
    top[0]->Swap(*batch->data_);
  } else {
    top[0]->safe_reshape_mode(true);
    top[0]->CopyDataFrom(*batch->data_, true, batch->data_packing(),
        this->transform_param_.forward_packing());
  }
  if (this->output_labels_) {
    top[1]->Swap(*batch->label_);
  }
  batch->set_id((size_t) -1);
  prefetches_free_[next_batch_queue_]->push(batch);
  next_batch_queue();
}

INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(BasePrefetchingDataLayer);

}  // namespace caffe
