#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/solver.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
ImageDataLayer<Ftype, Btype>::ImageDataLayer(const LayerParameter& param, size_t solver_rank)
    : BasePrefetchingDataLayer<Ftype, Btype>(param, solver_rank) {}

template <typename Ftype, typename Btype>
ImageDataLayer<Ftype, Btype>::~ImageDataLayer<Ftype, Btype>() {
  if (layer_inititialized_flag_.is_set()) {
    this->StopInternalThread();
  }
}

template <typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const int short_side = this->layer_param_.image_data_param().short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  size_t skip = 0UL;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    if (Caffe::gpus().size() > 1) {
      LOG(WARNING) << "Skipping data points is not supported in multiGPU mode";
    } else {
      skip = caffe_rng_rand() % this->layer_param_.image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points";
      CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    }
  }
  line_ids_.resize(this->threads_num());
  for (size_t i = 0; i < this->threads_num(); ++i) {
    line_ids_[i] = this->rank_ * this->threads_num() + i + skip;
  }

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  if (this->rank_ == 0) {
    // Read the file with filenames and labels
    ImageDataLayer<Ftype, Btype>::lines_.clear();
    const string &source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    string filename;
    int label;
    while (infile >> filename >> label) {
      ImageDataLayer<Ftype, Btype>::lines_.emplace_back(std::make_pair(filename, label));
    }
    if (this->layer_param_.image_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
      ShuffleImages();
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[line_ids_[0]].first,
      new_height, new_width, is_color, short_side);
  CHECK(cv_img.data) << "Could not load " << lines_[line_ids_[0]].first;
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  const int crop_height = crop <= 0 ? cv_img.rows : std::min(cv_img.rows, crop);
  const int crop_width = crop <= 0 ? cv_img.cols : std::min(cv_img.cols, crop);
  vector<int> top_shape { batch_size, cv_img.channels(), crop_height, crop_width };
  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ", "
      << top[0]->channels() << ", " << top[0]->height() << ", "
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->batch_transformer_->reshape(top_shape, label_shape);
  layer_inititialized_flag_.set();
}

template <typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template<typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::InitializePrefetch() {}

template <typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t) {
  CHECK(batch->data_->count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int short_side = image_data_param.short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  size_t line_id = line_ids_[thread_id];
  const size_t line_bucket = Caffe::gpus().size() * this->threads_num();
  const size_t lines_size = lines_.size();
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[line_id].first,
      new_height, new_width, is_color, short_side);
  CHECK(cv_img.data) << "Could not load " << lines_[line_id].first;
  int crop_height = crop;
  int crop_width = crop;
  if (crop <= 0) {
    LOG(INFO) << "Crop is not set. Using '" << (root_folder + lines_[line_id].first)
              << "' as a model, w=" << cv_img.rows << ", h=" << cv_img.cols;
    crop_height = cv_img.rows;
    crop_width = cv_img.cols;
  }

  // Infer the expected blob shape from a cv_img.
  vector<int> top_shape { batch_size, cv_img.channels(), crop_height, crop_width };
  batch->data_->Reshape(top_shape);
  vector<int> label_shape(1, batch_size);
  batch->label_->Reshape(label_shape);
  vector<Btype> tmp(top_shape[1] * top_shape[2] * top_shape[3]);

  Btype* prefetch_data = batch->data_->mutable_cpu_data<Btype>();
  Btype* prefetch_label = batch->label_->mutable_cpu_data<Btype>();
  Packing packing = NHWC;

  // datum scales
  const size_t buf_len = batch->data_->offset(1);
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    CHECK_GT(lines_size, line_id);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[line_id].first,
        new_height, new_width, is_color, short_side);
    if (cv_img.data) {
      int offset = batch->data_->offset(item_id);
#if defined(USE_CUDNN)
      this->dt(thread_id)->Transform(cv_img, prefetch_data + offset, buf_len, false);
#else
      CHECK_EQ(buf_len, tmp.size());
      this->dt(thread_id)->Transform(cv_img, prefetch_data + offset, buf_len, false);
      hwc2chw(top_shape[1], top_shape[3], top_shape[2], tmp.data(), prefetch_data + offset);
      packing = NCHW;
#endif
      prefetch_label[item_id] = lines_[line_id].second;
    }
    // go to the next iter
    line_ids_[thread_id] += line_bucket;
    if (line_ids_[thread_id] >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << this->print_current_device() << "Restarting data prefetching from start.";
      while (line_ids_[thread_id] >= lines_size) {
        line_ids_[thread_id] -= lines_size;
      }
      if (thread_id == 0 && this->rank_ == 0 && this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
    line_id = line_ids_[thread_id];
  }
  batch->set_data_packing(packing);
  batch->set_id(this->batch_id(thread_id));
}

INSTANTIATE_CLASS_CPU_FB(ImageDataLayer);
REGISTER_LAYER_CLASS_R(ImageData);

}  // namespace caffe
