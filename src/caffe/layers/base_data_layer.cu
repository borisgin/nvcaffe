#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void BasePrefetchingDataLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {

  cudaStream_t stream = Caffe::thread_stream();

  // Note: this function runs in one thread per object and one object per one Solver thread
//  shared_ptr<Batch> batch = pfull_.pop("Data layer prefetch queue empty");
  shared_ptr<Batch> batch = prefetches_full_[next_batch_queue_]->pop("Data layer prefetch queue empty");
//  if (batch->data_packing() == this->transform_param_.forward_packing()
//      && top[0]->shape() == batch->data_->shape()) {
//    top[0]->Swap(*batch->data_);
//  } else {
//    top[0]->safe_reshape_mode(true);

//  GPUMemory::Workspace tmp_gpu_holder(sizeof(Btype) * batch->data_->count(), Caffe::current_device());

//  CUDA_CHECK(cudaMemcpyAsync(tmp_gpu_holder.data(),
//      batch->data_->cpu_data<Btype>(), tmp_gpu_holder.size(), cudaMemcpyHostToDevice, stream));

//  TBlob<Btype> tmp(batch->data_->shape());

  tmp_.safe_reshape_mode(true);
  tmp_.Reshape(batch->data_->shape());
  tmp_.set_cpu_data(batch->data_->template mutable_cpu_data<Btype>());

  top[0]->CopyDataFrom(tmp_, //*batch->data_,
      true, batch->data_packing(),
        this->transform_param_.forward_packing());
//  }
  if (this->output_labels_) {
    top[1]->Swap(*batch->label_);
  }

//  LOG(INFO) << batch->data_->to_string();

//  LOG(INFO) << this->print_current_device() << " &&&&&& " << this << " " << batch->id();

      batch->set_id((size_t) -1L);
//  pfree_.push(batch);
  prefetches_free_[next_batch_queue_]->push(batch);
  next_batch_queue();
}

INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(BasePrefetchingDataLayer);

}  // namespace caffe
