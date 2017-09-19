#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>
#include <mutex>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template<typename Ftype, typename Btype>
class BaseDataLayer : public Layer<Ftype, Btype> {
 public:
  BaseDataLayer(const LayerParameter& param, size_t transf_num);
  virtual ~BaseDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  virtual void DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;

  // Data layers should be shared by multiple solvers in parallel
  bool ShareInParallel() const override { return true; }

  // Data layers have no bottoms, so reshaping is trivial.
  void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) override {}

  void Backward_cpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override {}
  void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom) override {}

 protected:
  TransformationParameter transform_param_;
  bool output_labels_;
};

template<typename Ftype>
class Batch {
 public:
  TBlob<Ftype> data_, label_;

  Batch() : id_((size_t) -1) {}
  ~Batch() {}

  size_t id() const {
    return id_;
  }

  void set_id(size_t id) {
    id_ = id;
  }

  size_t bytes() const {
    return sizeof(Ftype) * (data_.count() + label_.count());
  }

 private:
  size_t id_;
};

template<typename Ftype>
inline bool operator<(const shared_ptr <Batch<Ftype>>& a, const shared_ptr <Batch<Ftype>>& b) {
  return a->id() < b->id();
}

template<typename Ftype>
inline bool operator==(const shared_ptr <Batch<Ftype>>& a, const shared_ptr <Batch<Ftype>>& b) {
  return a->id() == b->id();
}

template<typename Ftype>
inline bool operator>(const shared_ptr <Batch<Ftype>>& a, const shared_ptr <Batch<Ftype>>& b) {
  return a->id() > b->id();
}

template<typename Ftype, typename Btype>
class BasePrefetchingDataLayer : public BaseDataLayer<Ftype, Btype>, public InternalThread {
 public:
  using DT = DataTransformer<Ftype>;
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  virtual ~BasePrefetchingDataLayer();
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_cpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;
  void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) override;

  DT* dt(int id) {
    return data_transformers_.at(id).get();
  }

 protected:
  void InternalThreadEntry() override;
  void InternalThreadEntryN(size_t thread_id) override;
  void ResizeQueues();
  void AllocatePrefetch();

  virtual void InitializePrefetch();
  virtual void load_batch(Batch<Ftype>* batch, int thread_id, size_t queue_id) = 0;
  virtual void start_reading() = 0;
  virtual size_t queue_id(size_t thread_id) const {
    return thread_id;
  }
  virtual bool auto_mode() const {
    return auto_mode_;
  }

  size_t batch_id(int thread_id) {
    size_t id = batch_ids_[thread_id];
    batch_ids_[thread_id] += this->threads_num();
    return id;
  }

  void next_batch_queue() {
    // spinning the wheel to the next queue:
    ++next_batch_queue_;
    if (next_batch_queue_ >= queues_num_) {
      next_batch_queue_ = 0;
    }
  }

  static size_t threads(const LayerParameter& param);
  static size_t parser_threads(const LayerParameter& param);
  static bool auto_mode(const LayerParameter& param);

  std::vector<size_t> batch_ids_;
  std::vector<shared_ptr<Batch<Ftype>>> prefetch_;
  const bool auto_mode_;
  size_t parsers_num_, transf_num_, queues_num_;
  std::vector<shared_ptr<BlockingQueue<shared_ptr<Batch<Ftype>>>>> prefetches_full_;
  std::vector<shared_ptr<BlockingQueue<shared_ptr<Batch<Ftype>>>>> prefetches_free_;
  size_t next_batch_queue_;
  // These two are for delayed init only
  std::vector<Blob*> bottom_init_;
  std::vector<Blob*> top_init_;

  vector<shared_ptr<DT>> data_transformers_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
