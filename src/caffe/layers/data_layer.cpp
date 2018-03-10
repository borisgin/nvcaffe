#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
DataLayer<Ftype, Btype>::DataLayer(const LayerParameter& param, size_t solver_rank)
  : BasePrefetchingDataLayer<Ftype, Btype>(param, solver_rank),
    cache_(param.data_param().cache()),
    shuffle_(param.data_param().shuffle()) {
  sample_only_.store(this->auto_mode_);
  init_offsets();
  datum_encoded_ = false;
}

template<typename Ftype, typename Btype>
void
DataLayer<Ftype, Btype>::init_offsets() {
  CHECK_EQ(this->transf_num_, this->threads_num());
  CHECK_LE(parser_offsets_.size(), this->transf_num_);
  CHECK_LE(queue_ids_.size(), this->transf_num_);
  parser_offsets_.resize(this->transf_num_);
  random_vectors_.resize(this->transf_num_);
  queue_ids_.resize(this->transf_num_);
  for (size_t i = 0; i < this->transf_num_; ++i) {
    parser_offsets_[i] = 0;
    queue_ids_[i] = i * this->parsers_num_;
    if (!random_vectors_[i]) {
      random_vectors_[i] = make_shared<TBlob<unsigned int>>();
    }
  }
}

template<typename Ftype, typename Btype>
DataLayer<Ftype, Btype>::~DataLayer() {
  this->StopInternalThread();
}

template<typename Ftype, typename Btype>
void
DataLayer<Ftype, Btype>::InitializePrefetch() {
  if (layer_inititialized_flag_.is_set()) {
    return;
  }
  const bool auto_mode = this->auto_mode_;
  if (auto_mode) {
    // Here we try to optimize memory split between prefetching and convolution.
    // All data and parameter blobs are allocated at this moment.
    // Now let's find out what's left...
    Net* pnet = this->parent_net();
    const size_t batch_bytes = pnet->prefetch_bytes<Ftype, Btype>();
    const size_t gpu_bytes = Caffe::min_avail_device_memory();
    const size_t batches_fit = gpu_bytes / batch_bytes;
    size_t parsers_num = this->parsers_num_;
    size_t transf_num = this->threads_num();
    if (this->is_gpu_transform()) {
      // in this mode memory demand is O(n) high
      size_t max_parsers_num = 2;
      const size_t max_transf_num = 4;
      float ratio = datum_encoded_ ? 3.F : 4.F;
      const float fit = std::min(float(max_parsers_num * max_transf_num),
          std::floor(batches_fit / ratio) - 1.F);
      parsers_num = std::min(max_parsers_num, std::max(1UL,
          static_cast<size_t>(std::sqrt(fit))));
      if (cache_ && parsers_num > 1UL) {
        LOG(INFO) << this->print_current_device() << " Reduced parser threads count from "
                  << parsers_num << " to 1 because cache is used";
        parsers_num = 1UL;
      }
      transf_num = std::min(max_transf_num, std::max(transf_num,
          static_cast<size_t>(std::lround(fit / parsers_num))));
      if (parsers_num > 1 && transf_num == max_transf_num - 1) {
        parsers_num = 1;
        transf_num = max_transf_num;
      }
      if (parsers_num == 2 && transf_num == 2) {
        parsers_num = 1;
        transf_num = max_transf_num;
      }
    } else {
      // in this mode memory demand is O(1)
      if (batches_fit > 0) {
        parsers_num = cache_ ? 1 : 3;
        transf_num = 4;
      }
    }

    this->RestartAllThreads(transf_num, true, false, Caffe::next_seed());
    this->transf_num_ = this->threads_num();
    this->parsers_num_ = parsers_num;
    this->queues_num_ = this->transf_num_ * this->parsers_num_;
    this->batch_transformer_->ResizeQueues(this->queues_num_);
    BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch();
    if (this->parsers_num_ > 1) {
      parser_offsets_[0]++;  // 0th already processed
    }
    this->auto_mode_ = false;
    layer_inititialized_flag_.set();
    this->go();  // kick off new threads if any
  }

  CHECK_EQ(this->threads_num(), this->transf_num_);
  LOG(INFO) << this->print_current_device() << " Parser threads: "
      << this->parsers_num_ << (auto_mode ? " (auto)" : "");
  LOG(INFO) << this->print_current_device() << " Transformer threads: "
      << this->transf_num_ << (auto_mode ? " (auto)" : "");
  layer_inititialized_flag_.set();
}

template<typename Ftype, typename Btype>
size_t DataLayer<Ftype, Btype>::queue_id(size_t thread_id) const {
  const size_t qid = queue_ids_[thread_id] + parser_offsets_[thread_id];
  parser_offsets_[thread_id]++;
  if (parser_offsets_[thread_id] >= this->parsers_num_) {
    parser_offsets_[thread_id] = 0UL;
    queue_ids_[thread_id] += this->parsers_num_ * this->threads_num();
  }
  return qid % this->queues_num_;
};

template<typename Ftype, typename Btype>
void
DataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const LayerParameter& param = this->layer_param();
  const int batch_size = param.data_param().batch_size();
  const bool cache = cache_ && this->phase_ == TRAIN;
  const bool shuffle = cache && shuffle_ && this->phase_ == TRAIN;

  if (this->auto_mode_) {
    if (!sample_reader_) {
      sample_reader_ = std::make_shared<DataReader<Datum>>(param, Caffe::solver_count(),
          this->rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          true,
          false,
          cache,
          shuffle,
          false);
    } else if (!reader_) {
      reader_ = std::make_shared<DataReader<Datum>>(param,
          Caffe::solver_count(),
          this->rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          false,
          true,
          cache,
          shuffle,
          this->phase_ == TRAIN);
    }
  } else if (!reader_) {
    reader_ = std::make_shared<DataReader<Datum>>(param,
        Caffe::solver_count(),
        this->rank_,
        this->parsers_num_,
        this->threads_num(),
        batch_size,
        false,
        false,
        cache,
        shuffle,
        this->phase_ == TRAIN);
    start_reading();
  }
  // Read a data point, and use it to initialize the top blob.
  shared_ptr<Datum> sample_datum = sample_only_ ? sample_reader_->sample() : reader_->sample();
  datum_encoded_ = sample_datum->encoded();
  this->ResizeQueues();
  init_offsets();

  // Reshape top[0] and prefetch_data according to the batch_size.
  // Note: all these reshapings here in load_batch are needed only in case of
  // different datum shapes coming from database.
  Packing packing = NHWC;  // OpenCV
  vector<int> top_shape = this->bdt(0)->Transform(sample_datum.get(), nullptr, 0, packing);
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  if (this->is_gpu_transform()) {
    CHECK(Caffe::mode() == Caffe::GPU);
    LOG(INFO) << this->print_current_device() << " Transform on GPU enabled";
    tmp_gpu_buffer_.resize(this->threads_num());
    for (int i = 0; i < this->tmp_gpu_buffer_.size(); ++i) {
      this->tmp_gpu_buffer_[i] = make_shared<GPUMemory::Workspace>();
    }
  }
  // label
  vector<int> label_shape(1, batch_size);
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
  }
  this->batch_transformer_->reshape(top_shape, label_shape, this->is_gpu_transform());
  LOG(INFO) << this->print_current_device() << " Output data size: "
      << top[0]->num() << ", "
      << top[0]->channels() << ", "
      << top[0]->height() << ", "
      << top[0]->width();
}

template<typename Ftype, typename Btype>
void DataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t queue_id) {
  const bool sample_only = sample_only_.load();
  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();

  const size_t qid = sample_only ? 0UL : queue_id;
  DataReader<Datum>* reader = sample_only ? sample_reader_.get() : reader_.get();
  shared_ptr<Datum> init_datum = reader->full_peek(qid);
  CHECK(init_datum);
  const bool use_gpu_transform = this->is_gpu_transform();
  Packing packing = NHWC;  // OpenCV
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape =
      this->bdt(thread_id)->Transform(init_datum.get(), nullptr, 0, packing);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  if (top_shape != batch->data_->shape()) {
    batch->data_->Reshape(top_shape);
  }
  int init_datum_height = init_datum->height();
  int init_datum_width = init_datum->width();
  const int color_mode = this->transform_param_.force_color() ?
                         1 : (this->transform_param_.force_gray() ? -1 : 0);
  size_t datum_sizeof_element = 0UL;
  int datum_len = top_shape[1] * top_shape[2] * top_shape[3];
  size_t datum_size = 0UL;
  const char *src_ptr = nullptr;
  vector<char> src_buf;
  cv::Mat img;
  if (use_gpu_transform) {
    if (init_datum->encoded()) {
      DecodeDatumToCVMat(*init_datum, color_mode, img, false, false);
      datum_len = img.channels() * img.rows * img.cols;
      datum_sizeof_element = sizeof(char);
      init_datum_height = img.rows;
      init_datum_width = img.cols;
    } else {
      datum_len = init_datum->channels() * init_datum->height() * init_datum->width();
      CHECK_GT(datum_len, 0);
      const string &datum_data = init_datum->data();
      if (datum_data.empty()) {
        CHECK_LE(sizeof(float), sizeof(Ftype));
        datum_sizeof_element = sizeof(float);
      } else {
        CHECK_LE(sizeof(uint8_t), sizeof(Ftype));
        CHECK_EQ(datum_len, datum_data.size());
        datum_sizeof_element = sizeof(uint8_t);
      }
    }

    vector<int> random_vec_shape(1, batch_size * 3);
    random_vectors_[thread_id]->Reshape(random_vec_shape);
    datum_size = datum_len * datum_sizeof_element;
    src_buf.resize(datum_size);
  }
  if (this->output_labels_) {
    batch->label_->Reshape(vector<int>(1, batch_size));
  }
  Ftype* top_label = this->output_labels_ ?
      batch->label_->template mutable_cpu_data_c<Ftype>(false) : nullptr;

  void* dst_gptr = nullptr;
  Btype* dst_cptr = nullptr;
  if (use_gpu_transform) {
    size_t buffer_size = top_shape[0] * top_shape[1] * init_datum_height * init_datum_width;
    tmp_gpu_buffer_[thread_id]->safe_reserve(buffer_size);
    dst_gptr = tmp_gpu_buffer_[thread_id]->data();
  } else {
    dst_cptr = batch->data_->template mutable_cpu_data_c<Btype>(false);
  }

  size_t current_batch_id = 0UL;
  const size_t buf_len = batch->data_->offset(1);
  for (size_t entry = 0; entry < batch_size; ++entry) {
    shared_ptr<Datum> datum = reader->full_pop(qid, "Waiting for datum");
    size_t item_id = datum->record_id() % batch_size;
    if (item_id == 0UL) {
      current_batch_id = datum->record_id() / batch_size;
    }
    // Copy label.
    if (top_label != nullptr) {
      top_label[item_id] = datum->label();
    }

    if (use_gpu_transform) {
      cudaStream_t stream = Caffe::thread_stream(Caffe::GPU_TRANSF_GROUP);
      if (datum->encoded()) {
        DecodeDatumToSignedBuf(*datum, color_mode, src_buf.data(), datum_size, false);
      } else {
        CHECK_EQ(datum_len, datum->channels() * datum->height() * datum->width())
          << "Datum size can't vary in the same batch";
        src_ptr = datum->data().size() > 0 ?
                  &datum->data().front() :
                  reinterpret_cast<const char*>(&datum->float_data().Get(0));
        // NOLINT_NEXT_LINE(caffe/alt_fn)
        std::memcpy(src_buf.data(), src_ptr, datum_size);
      }
      CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst_gptr) + item_id * datum_size,
          src_buf.data(), datum_size, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      this->bdt(thread_id)->Fill3Randoms(&random_vectors_[thread_id]->
          mutable_cpu_data()[item_id * 3]);
    } else {
      // Get data offset for this datum to hand off to transform thread
      const size_t offset = batch->data_->offset(item_id);
      CHECK_EQ(0, offset % buf_len);
#if defined(USE_CUDNN)
      vector<int> shape = this->bdt(thread_id)->Transform(datum.get(), dst_cptr + offset,
          buf_len, packing, false);
#else
      vector<Btype> tmp(top_shape[1] * top_shape[2] * top_shape[3]);
      CHECK_EQ(buf_len, tmp.size());
      vector<int> shape = this->bdt(thread_id)->Transform(datum.get(), tmp.data(), buf_len,
          packing, false);
      if (packing == NHWC) {
        hwc2chw(top_shape[1], top_shape[3], top_shape[2], tmp.data(), dst_cptr + offset);
        packing = NCHW;
      } else {
        // NOLINT_NEXT_LINE(caffe/alt_fn)
        memcpy(dst_cptr + offset, tmp.data(), buf_len * sizeof(Btype));
      }
#endif
      CHECK_EQ(top_shape[1], shape[1]) << "Number of channels can't vary in the same batch";
      CHECK_EQ(top_shape[2], shape[2]) << "Image height can't vary in the same batch";
      CHECK_EQ(top_shape[3], shape[3]) << "Image width can't vary in the same batch";
    }
    reader->free_push(qid, datum);
  }

  if (use_gpu_transform) {
    this->fdt(thread_id)->TransformGPU(top_shape[0], top_shape[1],
        init_datum_height,  // non-crop
        init_datum_width,  // non-crop
        datum_sizeof_element,
        dst_gptr,
        batch->data_->template mutable_gpu_data_c<Ftype>(false),
        random_vectors_[thread_id]->gpu_data(), true);
    packing = NCHW;
  }

  batch->set_data_packing(packing);
  batch->set_id(current_batch_id);
  sample_only_.store(false);
}

INSTANTIATE_CLASS_FB(DataLayer);
REGISTER_LAYER_CLASS_R(Data);

}  // namespace caffe
