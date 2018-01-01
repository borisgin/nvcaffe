#ifndef CAFFE_BATCH_TRANSFORMER_CPP_H
#define CAFFE_BATCH_TRANSFORMER_CPP_H

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

class Batch {
 public:
  shared_ptr<Blob> data_;
  shared_ptr<Blob> label_;

  Batch(Type data_type, Type diff_type)
      : data_(Blob::create(data_type, diff_type)), label_(Blob::create(data_type, diff_type)),
        id_((size_t) -1), data_packing_(NCHW) {}

  size_t id() const {
    return id_;
  }
  void set_id(size_t id) {
    id_ = id;
  }
  size_t bytes() const {
    return data_->sizeof_data() + label_->sizeof_data();
  }
  Packing data_packing() const {
    return data_packing_;
  }
  void set_data_packing(Packing packing) {
    data_packing_ = packing;
  }

  DISABLE_COPY_MOVE_AND_ASSIGN(Batch);

 private:
  size_t id_;
  Packing data_packing_;
};


template<typename Ftype, typename Btype>
class BatchTransformer : public InternalThread {
  typedef BlockingQueue<boost::shared_ptr<Batch>> BBQ;

 public:
  BatchTransformer(int target_device, size_t rank_, size_t queues_num,
      const TransformationParameter& transform_param, bool gpu_transform);

  shared_ptr<Batch> prefetched_pop_free(size_t qid) {
    return this->prefetches_free_[qid]->pop();
  }

  void prefetched_push_full(size_t qid, const shared_ptr<Batch> &batch) {
    prefetches_full_[qid]->push(batch);
  }

  boost::shared_ptr<Batch> processed_pop() {
    return this->processed_full_.pop();
  }

  void processed_push(const boost::shared_ptr<Batch>& batch) {
    return this->processed_free_.push(batch);
  }

  void reshape(const vector<int>& data_shape, const vector<int>& label_shape,
      bool preallocate = false);

  size_t prefetch_bytes() const {
    return this->prefetch_[0]->bytes();
  }

  void ResizeQueues(size_t queues_num);

 protected:
  void InternalThreadEntry() override;

  size_t queues_num_, next_batch_queue_;
  const TransformationParameter transform_param_;
  const bool gpu_transform_;

  std::vector<boost::shared_ptr<Batch>> prefetch_;
  std::vector<boost::shared_ptr<BBQ>> prefetches_full_;
  std::vector<boost::shared_ptr<BBQ>> prefetches_free_;

  void next_batch_queue() {
    ++next_batch_queue_;
    if (next_batch_queue_ >= queues_num_) {
      next_batch_queue_ = 0;
    }
  }

  void resize(bool skip_to_next);

 private:
  BBQ processed_full_;
  BBQ processed_free_;
  TBlob<Btype> tmp_;
};

}

#endif //CAFFE_BATCH_TRANSFORMER_CPP_H
