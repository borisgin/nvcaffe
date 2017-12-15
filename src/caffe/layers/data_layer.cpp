#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
DataLayer<Ftype, Btype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Ftype, Btype>(param),
    cache_(param.data_param().cache()),
    shuffle_(param.data_param().shuffle()) {
  sample_only_.store(this->auto_mode_ && this->phase_ == TRAIN);
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
  if (this->auto_mode()) {
    this->AllocatePrefetch();
//    P2PManager::dl_bar_wait();
    // Here we try to optimize memory split between prefetching and convolution.
    // All data and parameter blobs are allocated at this moment.
    // Now let's find out what's left...
    size_t current_parsers_num = this->parsers_num_;
    size_t current_transf_num = this->threads_num();
#ifndef CPU_ONLY
    Net* pnet = this->parent_net();
    const size_t batch_bytes = pnet->prefetch_bytes<Ftype, Btype>();
    size_t gpu_bytes = Caffe::min_avail_device_memory();
    size_t batches_fit = gpu_bytes / batch_bytes;
#else
    size_t batches_fit = this->queues_num_;
#endif
    size_t max_parsers_num = 2;
    const size_t max_transf_num = 4;
    float ratio = datum_encoded_ ? 3.F : 4.F;
    if (pnet != nullptr) {
      Solver* psolver = pnet->parent_solver();
      if (psolver != nullptr) {
        if (pnet->layers().size() < 100) {
          ratio = 2.F; // 1:2 for "i/o bound", 1:4 or 1:3 otherwise
        }
      }
    }
    const float fit = std::min(float(max_parsers_num * max_transf_num),
        std::floor(batches_fit / ratio));
    current_parsers_num = std::min(max_parsers_num, std::max(1UL,
        static_cast<size_t>(std::sqrt(fit))));
    if (cache_ && current_parsers_num > 1UL) {
      LOG(INFO) << this->print_current_device() << " Reduced parser threads count from "
                << current_parsers_num << " to 1 because cache is used";
      current_parsers_num = 1UL;
    }
    current_transf_num = std::min(max_transf_num, std::max(current_transf_num,
        static_cast<size_t>(std::lround(fit / current_parsers_num))));
    if (current_parsers_num > 1 && current_transf_num == max_transf_num - 1) {
      current_parsers_num = 1;
      current_transf_num = max_transf_num;
    }
    pnet->set_mins(current_parsers_num, current_transf_num);
    P2PManager::dl_bar_wait();
    {
      std::lock_guard<std::mutex> lock(mutex_init_);
      // preventing different number of threads on different GPUs
      current_parsers_num = pnet->min_parsers();
      current_transf_num = pnet->min_transformers();
    }
    this->RestartAllThreads(current_transf_num, true, false, Caffe::next_seed());
    this->transf_num_ = this->threads_num();
    this->parsers_num_ = current_parsers_num;
    this->queues_num_ = this->transf_num_ * this->parsers_num_;

    this->batch_transformer_->ResizeQueues(this->queues_num_);

//      this->qbar_.reset(new boost::barrier(this->transf_num_));
//      this->lbar_.reset(new boost::barrier(this->transf_num_));

    BasePrefetchingDataLayer<Ftype, Btype>::InitializePrefetch();
//      if (current_transf_num > 1) {
//        this->batch_transformer_->next_batch_queue();  // 0th already processed
//      }
    if (this->parsers_num_ > 1) {
      parser_offsets_[0]++;  // 0th already processed
    }
    this->go();  // kick off new threads if any
//      this->batch_transformer_ = make_shared<BatchTransformer<Ftype, Btype>>(Caffe::current_device(),
//          this->solver_rank_, this->queues_num_, this->transform_param_);

  }

  CHECK_EQ(this->threads_num(), this->transf_num_);
  LOG(INFO) << this->print_current_device() << " Parser threads: "
      << this->parsers_num_ << (this->auto_mode_ ? " (auto)" : "");
  LOG(INFO) << this->print_current_device() << " Transformer threads: "
      << this->transf_num_ << (this->auto_mode_ ? " (auto)" : "");
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
  const bool use_gpu_transform = this->is_gpu_transform();
  const bool cache = cache_ && this->phase_ == TRAIN;
  const bool shuffle = cache && shuffle_ && this->phase_ == TRAIN;

  if (this->auto_mode_) {
    if (!sample_reader_) {
      sample_reader_ = make_shared<DataReader>(param,
          Caffe::solver_count(),
          this->solver_rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          true,
          false,
          cache,
          shuffle);
    } else if (!reader_) {
      reader_ = make_shared<DataReader>(param,
          Caffe::solver_count(),
          this->solver_rank_,
          this->parsers_num_,
          this->threads_num(),
          batch_size,
          false,
          true,
          cache,
          shuffle);
    }
  } else if (!reader_) {
    reader_ = make_shared<DataReader>(param,
        Caffe::solver_count(),
        this->solver_rank_,
        this->parsers_num_,
        this->threads_num(),
        batch_size,
        false,
        false,
        cache,
        shuffle);
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
  vector<int> top_shape = this->dt(0)->template Transform<Btype>(sample_datum.get(),
      nullptr, 0, packing);
  top_shape[0] = batch_size;
  top[0]->safe_reshape_mode(true);
  top[0]->Reshape(top_shape);

//  vector<int> random_vec_shape(1, batch_size * 3);
//  LOG(INFO) << this->print_current_device() << " ReshapePrefetch "
//      << top_shape[0] << ", "
//      << top_shape[1] << ", "
//      << top_shape[2] << ", "
//      << top_shape[3];
//  for (int i = 0; i < this->prefetch_.size(); ++i) {
//    this->prefetch_[i]->data_->Reshape(top_shape);
//  }
  if (use_gpu_transform) {
    LOG(INFO) << this->print_current_device() << " Transform on GPU enabled";
  }
  // label
  vector<int> label_shape(1, batch_size);
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
//    for (int i = 0; i < this->prefetch_.size(); ++i) {
//      this->prefetch_[i]->label_->Reshape(label_shape);
//    }
  }
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
  DataReader* reader = sample_only ? sample_reader_.get() : reader_.get();
  shared_ptr<Datum> init_datum = reader->full_peek(qid);
  CHECK(init_datum);
  const bool use_gpu_transform = this->is_gpu_transform();
  Packing packing = NHWC;  // OpenCV
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape =
      this->dt(thread_id)->template Transform<Btype>(init_datum.get(), nullptr, 0, packing);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  if (top_shape != batch->data_->shape()) {
    batch->data_->Reshape(top_shape);
  }
#ifndef CPU_ONLY
  int init_datum_height = init_datum->height();
  int init_datum_width = init_datum->width();
  const int color_mode = this->transform_param_.force_color() ?
                         1 : (this->transform_param_.force_gray() ? -1 : 0);
  size_t datum_sizeof_element = 0UL;
  int datum_len = top_shape[1] * top_shape[2] * top_shape[3];
  size_t datum_size = 0UL;
  cudaStream_t stream = Caffe::thread_stream();
  const char *src_ptr = nullptr;
  size_t src_buf_pos = 0UL;
  size_t src_buf_items = 0UL;
  size_t src_buf_size = 0UL;
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
    // sqrt(batch) buckets
    src_buf_items = (size_t) std::lround(std::sqrt((double)batch_size));
    src_buf_size = src_buf_items * datum_size;
    src_buf.resize(src_buf_size);
//    if (init_datum->encoded()) {
//      vector<int> init_shape { top_shape[0], top_shape[1], init_datum_height, init_datum_width };
//      batch->data_->Reshape(init_shape);
//      dst_gptr = batch->data_->template mutable_gpu_data_c<Ftype>(false);
//    }
//    dst_gptr = tmp_gpu_holder_[thread_id]->data();
  }
  size_t last_item_id = 0UL;

//if (Caffe::current_device() == 0)
//  LOG(INFO) << this->print_current_device() << " ######## " << top_->size()
//      << " " << (*top_)[0]->count() << " " << (*top_)[1]->count();
//////      << " bottom " << bottom_->size()
//////      << " " << (*bottom_)[0]->count() << " " << (*bottom_)[1]->count();






#endif
//  size_t holder_size = //init_datum->encoded() ? sizeof(Btype) *
////      top_shape[0] * top_shape[1] * init_datum_height * init_datum_width :
//      sizeof(Btype) * batch->data_->count();
//
//  // todo?
////  if (!use_gpu_transform && tmp_cpu_holder_[thread_id].size() < holder_size) {
////    tmp_cpu_holder_[thread_id].resize(holder_size);
////  }
//
//  if (thread_id == 0) {
//    top_->at(0)->Reshape(top_shape);
//    top_->at(0)->mutable_gpu_data_c<Btype>(false);
////    tmp_gpu_holder_[thread_id]->safe_reserve(holder_size);
    if (this->output_labels_) {
      batch->label_->Reshape(vector<int>(1, batch_size));
//      batch->label_->template mutable_cpu_data_c<Ftype>(false);
    }
//    batch->data_->template mutable_gpu_data_c<Ftype>(false);
//  }

//   LOG(INFO) << this->print_current_device() << " ######## " << this
//      << " " << thread_id << " " << qid;


//  this->lbar_->wait();

  Ftype* top_label = this->output_labels_ ?
      batch->label_->template mutable_cpu_data_c<Ftype>(false) : nullptr;
//
  Btype* dst_gptr = use_gpu_transform ? batch->data_->mutable_gpu_data_c<Btype>(false) : nullptr;

//  void* dst_gptr = use_gpu_transform ? tmp_gpu_holder_[thread_id]->data() : nullptr;
  Btype* dst_cptr =batch->data_->template mutable_cpu_data_c<Btype>(false);
      //use_gpu_transform ? nullptr : &tmp_cpu_holder_[thread_id].front();


  size_t current_batch_id = 0UL;
  const size_t buf_len = batch->data_->offset(1);
  for (size_t entry = 0; entry < batch_size; ++entry) {
    shared_ptr<Datum> datum = reader->full_pop(qid, "Waiting for datum");
    size_t item_id = datum->record_id() % batch_size;

//    size_t i = datum->record_id() % batch_size;
//    size_t b = i / this->queues_num_;
//    size_t item_id = b + qid;

////if (Caffe::current_device() == 0)
//        LOG(INFO) << this->print_current_device() << " ********** "
//            << datum->record_id()  << " qn=" << this->queues_num_
//            << " thread=" << thread_id  << " qid=" << qid
////////            << " " << i << " " << b
//            << " " << item_id;



    if (item_id == 0UL) {
//      if (thread_id == 0 && item_id == 0UL) {
      current_batch_id = datum->record_id() / batch_size;
    }
    // Copy label.
    if (top_label != nullptr) {
      top_label[item_id] = datum->label();
    }

    if (use_gpu_transform) {
#ifndef CPU_ONLY
      if (datum->encoded()) {
        DecodeDatumToSignedBuf(*datum, color_mode,
            &src_buf[src_buf_pos * datum_size], datum_size, false);
      } else {
        CHECK_EQ(datum_len, datum->channels() * datum->height() * datum->width())
          << "Datum size can't vary in the same batch";
        src_ptr = datum->data().size() > 0 ?
                  &datum->data().front() :
                  reinterpret_cast<const char*>(&datum->float_data().Get(0));
        std::memcpy(src_buf.data() + src_buf_pos * datum_size, src_ptr, datum_size);
      }
      ++src_buf_pos;
      if (src_buf_pos == src_buf_items) {
        src_buf_pos = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<char*>(dst_gptr) + last_item_id * datum_size,
            src_buf.data(), src_buf_size, cudaMemcpyHostToDevice, stream));
//        CUDA_CHECK(cudaStreamSynchronize(stream));
        last_item_id = item_id + 1;
      }
      this->dt(thread_id)->Fill3Randoms(&random_vectors_[thread_id]->
          mutable_cpu_data()[item_id * 3]);
#else
      NO_GPU;
#endif
    } else {
      // Get data offset for this datum to hand off to transform thread
      const size_t offset = batch->data_->offset(item_id);
      CHECK_EQ(0, offset % buf_len);
      vector<int> shape = this->dt(thread_id)->Transform(datum.get(), dst_cptr + offset,
          buf_len, packing, false);
      CHECK_EQ(top_shape[1], shape[1]) << "Number of channels can't vary in the same batch";
      CHECK_EQ(top_shape[2], shape[2]) << "Image height can't vary in the same batch";
      CHECK_EQ(top_shape[3], shape[3]) << "Image width can't vary in the same batch";
    }
    reader->free_push(qid, datum);
  }

//  const bool needs_repack = packing != this->transform_param_.forward_packing();
  if (use_gpu_transform) {
#ifndef CPU_ONLY
    if (src_buf_pos > 0) {
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<char*>(dst_gptr) + last_item_id * datum_size,
          src_buf.data(), src_buf_pos * datum_size, cudaMemcpyHostToDevice, stream));
//      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
//  else {
//    const size_t gpu_holder_size = sizeof(Btype) *
//        top_shape[0] * top_shape[1] * top_shape[2] * top_shape[3];
//
//   // LOG(INFO) << this->print_current_device() << " ********** " << top_->at(0)->to_string();
//
////    tmp_gpu_holder_[thread_id]->safe_reserve(gpu_holder_size);
////    //tmp_cpu_holder_[thread_id].size());
////    dst_gptr = tmp_gpu_holder_[thread_id]->data();
//
//    CUDA_CHECK(cudaMemcpyAsync(dst_gptr, dst_cptr, gpu_holder_size,
//        cudaMemcpyHostToDevice, stream));
//    //batch->data_->template mutable_gpu_data_c<Btype>(false);
//  }

////  if (needs_repack) {
////  void* repack_dst_gptr =
//    cudnnHandle_t handle = Caffe::cudnn_handle();
//    cudnnTensorDescriptor_t src_desc, dst_desc;
//    CUDNN_CHECK(cudnnCreateTensorDescriptor(&src_desc));
//    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dst_desc));
//    cudnn::setTensor4dDesc(&src_desc,
//        use_gpu_transform ? CUDNN_DATA_INT8 : cudnn_dt<Btype>(),
//        packing, batch->data_->shape());
//    cudnn::setTensor4dDesc(&dst_desc, cudnn_dt<Ftype>(), this->transform_param_.forward_packing(),
//        batch->data_->shape());
//
//    CUDNN_CHECK(cudnnTransformTensor(handle,
//        cudnn::one(tp<float>()),
//        src_desc, dst_gptr,
//        cudnn::zero(tp<Ftype>()),
//        dst_desc, batch->data_->template mutable_gpu_data_c<Ftype>(false)));
//        //tmp_gpu_holder_[thread_id]->data()));
//    CUDA_CHECK(cudaStreamSynchronize(stream));
//    CUDNN_CHECK(cudnnDestroyTensorDescriptor(src_desc));
//    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dst_desc));
//
////    dst_gptr = tmp_gpu_holder_[thread_id]->data();
////    batch->data_->Reshape(top_shape);
////    datum_sizeof_element = sizeof(Ftype);
//
//    packing = this->transform_param_.forward_packing();
////  }

  if (use_gpu_transform) {
    this->dt(thread_id)->TransformGPU(top_shape[0], top_shape[1],
        init_datum_height,  // non-crop
        init_datum_width,  // non-crop
        datum_sizeof_element,
        dst_gptr,
        batch->data_->template mutable_gpu_data_c<Ftype>(false),
        random_vectors_[thread_id]->gpu_data(), true); //needs_repack);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    packing = NCHW;
#else
    NO_GPU;
#endif
  }

  batch->set_data_packing(packing);
  batch->set_id(current_batch_id);
  sample_only_.store(false);
}

INSTANTIATE_CLASS_FB(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
