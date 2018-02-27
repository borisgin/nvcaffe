#include "caffe/batch_transformer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
BatchTransformer<Ftype, Btype>::BatchTransformer(int target_device, size_t rank_,
    size_t queues_num, const TransformationParameter& transform_param, bool gpu_transform) :
    InternalThread(target_device, rank_, 1, false),
    queues_num_(queues_num),
    next_batch_queue_(0UL),
    transform_param_(transform_param),
    gpu_transform_(gpu_transform) {
  shared_ptr<Batch> processed = make_shared<Batch>(tp<Ftype>(), tp<Ftype>());
  processed_free_.push(processed);
  resize(false);
  StartInternalThread();
}

template<typename Ftype, typename Btype>
void BatchTransformer<Ftype, Btype>::ResizeQueues(size_t queues_num) {
  StopInternalThread();
  queues_num_ = queues_num;
  if (queues_num_ > prefetches_free_.size()) {
    resize(true);
  }
  StartInternalThread(true);
}

template<typename Ftype, typename Btype>
void BatchTransformer<Ftype, Btype>::resize(bool skip_to_next) {
  size_t size = prefetches_free_.size();
  prefetches_free_.resize(queues_num_);
  prefetches_full_.resize(queues_num_);
  for (size_t i = size; i < queues_num_; ++i) {
    // prefetch is Btype (cpu transform) or Ftype (gpu_transform_), processed is Ftype
    shared_ptr<Batch> batch = gpu_transform_ ?
                              make_shared<Batch>(tp<Ftype>(), tp<Ftype>()) :
                              make_shared<Batch>(tp<Btype>(), tp<Btype>());
    prefetch_.push_back(batch);
    prefetches_free_[i] = make_shared<BlockingQueue<shared_ptr<Batch>>>();
    prefetches_full_[i] = make_shared<BlockingQueue<shared_ptr<Batch>>>();
    prefetches_free_[i]->push(batch);
  }
  if (skip_to_next) {
    next_batch_queue();  // 0th already processed
  }
}

template<typename Ftype, typename Btype>
void BatchTransformer<Ftype, Btype>::InternalThreadEntry() {
  try {
    while (!must_stop(0)) {
      shared_ptr<Batch> batch =
          prefetches_full_[next_batch_queue_]->pop("Data layer prefetch queue empty");
      boost::shared_ptr<Batch> top = processed_free_.pop();
      if (batch->data_->is_data_on_gpu() && top->data_->shape() == batch->data_->shape() &&
          batch->data_packing() == this->transform_param_.forward_packing()) {
        top->data_->Swap(*batch->data_);
      } else {
        if (batch->data_->is_data_on_gpu()) {
          top->data_->CopyDataFrom(*batch->data_, true,
              batch->data_packing(), transform_param_.forward_packing(), Caffe::GPU_TRANSF_GROUP);
        } else {
          if (tmp_.shape() != batch->data_->shape()) {
            tmp_.Reshape(batch->data_->shape());
          }
          if (top->data_->shape() != batch->data_->shape()) {
            top->data_->Reshape(batch->data_->shape());
          }
          tmp_.set_cpu_data(batch->data_->template mutable_cpu_data<Btype>());
          top->data_->CopyDataFrom(tmp_, false,
              batch->data_packing(), transform_param_.forward_packing(), Caffe::GPU_TRANSF_GROUP);
        }
      }
      top->label_->Swap(*batch->label_);
      processed_full_.push(top);
      batch->set_id((size_t) -1L);
      prefetches_free_[next_batch_queue_]->push(batch);
      next_batch_queue();
    }
  }catch (boost::thread_interrupted&) {
  }
}

template<typename Ftype, typename Btype>
void BatchTransformer<Ftype, Btype>::reshape(const vector<int>& data_shape,
    const vector<int>& label_shape, bool preallocate) {
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    prefetch_[i]->data_->Reshape(data_shape);
    prefetch_[i]->label_->Reshape(label_shape);
    if (preallocate && Caffe::mode() == Caffe::GPU) {
      prefetch_[i]->data_->template mutable_gpu_data_c<Ftype>(false);
      prefetch_[i]->label_->template mutable_gpu_data_c<Ftype>(false);
    }
  }
  shared_ptr<Batch> processed_batch;
  if (processed_free_.try_peek(&processed_batch)) {
    processed_batch->data_->Reshape(data_shape);
    processed_batch->label_->Reshape(label_shape);
    if (preallocate && Caffe::mode() == Caffe::GPU) {
      processed_batch->data_->template mutable_gpu_data_c<Ftype>(false);
      processed_batch->label_->template mutable_gpu_data_c<Ftype>(false);
    }
  }
}

INSTANTIATE_CLASS_FB(BatchTransformer);

}
