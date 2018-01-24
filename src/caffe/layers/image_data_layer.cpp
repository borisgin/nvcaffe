#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/solver.hpp"
//#include "caffe/data_transformer.hpp"
//#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
//#include "caffe/util/benchmark.hpp"
//#include "caffe/util/io.hpp"
//#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

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
    if (Caffe::device_count() > 1) {
      LOG(WARNING) << "Skipping data points is not supported in multiGPU mode";
    } else {
      skip = caffe_rng_rand() % this->layer_param_.image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points";
      CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    }
  }
  line_ids_.resize(this->threads_num());
  for (size_t i = 0; i < this->threads_num(); ++i) {
    line_ids_[i] = this->solver_rank_ * this->threads_num() + i + skip;
  }

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  if (this->solver_rank_ == 0) {
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
  }

//  if (this->layer_param_.image_data_param().shuffle()) {
//    // randomly shuffle data
//    LOG(INFO) << "Shuffling data";
//    prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
//    ShuffleImages();
//  }
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

//template <typename Ftype, typename Btype>
//void ImageDataLayer<Ftype, Btype>::ShuffleImages() {
//  if (Caffe::root_solver()) {
//    caffe::rng_t *prefetch_rng =
//        static_cast<caffe::rng_t *>(prefetch_rng_->generator());
//    shuffle(ImageDataLayer<Ftype, Btype>::lines_.begin(),
//            ImageDataLayer<Ftype, Btype>::lines_.end(), prefetch_rng);
//  }
//}

template<typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::InitializePrefetch() {}

// Borrowed this one to count line_id
//template<typename Ftype, typename Btype>
//size_t ImageDataLayer<Ftype, Btype>::queue_id(size_t thread_id) const {
//
//
//}

// This function is called on prefetch thread
template <typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t) {
//  CPUTimer batch_timer;
//  batch_timer.Start();
//  double read_time = 0;
//  double trans_time = 0;
//  CPUTimer timer;
  CHECK(batch->data_->count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int short_side = image_data_param.short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

//#if CV_VERSION_EPOCH == 2
//  cv::setNumThreads(0);  // cv::resize crashes otherwise
//#endif

  size_t line_id = line_ids_[thread_id];
  const size_t line_bucket = Caffe::device_count() * this->threads_num();
  const size_t lines_size = lines_.size();
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[line_id].first,
      new_height, new_width, is_color, short_side);
  CHECK(cv_img.data) << "Could not load " << lines_[line_id].first;
  const int crop_height = crop <= 0 ? cv_img.rows : std::min(cv_img.rows, crop);
  const int crop_width = crop <= 0 ? cv_img.cols : std::min(cv_img.cols, crop);
  // Infer the expected blob shape from a cv_img.
  vector<int> top_shape { batch_size, cv_img.channels(), crop_height, crop_width };
  batch->data_->Reshape(top_shape);
  vector<int> label_shape(1, batch_size);
  batch->label_->Reshape(label_shape);
  vector<Btype> tmp(top_shape[1] * top_shape[2] * top_shape[3]);

  Btype* prefetch_data = batch->data_->mutable_cpu_data<Btype>();
  Btype* prefetch_label = batch->label_->mutable_cpu_data<Btype>();
  Packing packing = NCHW; //NHWC;

  // datum scales
  const size_t buf_len = batch->data_->offset(1);
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
//    timer.Start();
    CHECK_GT(lines_size, line_id);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[line_id].first,
        new_height, new_width, is_color, short_side);
    CHECK(cv_img.data) << "Could not load " << lines_[line_id].first;
//    read_time += timer.MicroSeconds();
//    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_->offset(item_id);

//#if defined(USE_CUDNN)
//    this->dt(thread_id)->Transform(cv_img, prefetch_data + offset, buf_len, true);
//#else
    CHECK_EQ(buf_len, tmp.size());
    this->dt(thread_id)->Transform(cv_img, prefetch_data + offset, buf_len, false);
    hwc2chw(top_shape[1], top_shape[3], top_shape[2], tmp.data(), prefetch_data + offset);
//    packing = NCHW;
//#endif
//    trans_time += timer.MicroSeconds();
    prefetch_label[item_id] = lines_[line_id].second;

//    unsigned int Rand(int n) const

    // go to the next iter
    if (this->layer_param_.image_data_param().shuffle()) {
      const unsigned int rn = this->dt(thread_id)->Rand(lines_size / line_bucket + 1);
      line_ids_[thread_id] = rn * line_bucket;
    } else {
      line_ids_[thread_id] += line_bucket;
    }
    if (line_ids_[thread_id] >= lines_size) {
      line_ids_[thread_id] -= lines_size;
    }
    line_id = line_ids_[thread_id];
  }
//  batch_timer.Stop();
//  DLOG(INFO) << this->print_current_device()
//             << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//  DLOG(INFO) << this->print_current_device()
//             << "     Read time: " << read_time / 1000 << " ms.";
//  DLOG(INFO) << this->print_current_device()
//             << "Transform time: " << trans_time / 1000 << " ms.";

  batch->set_data_packing(packing);
  batch->set_id(this->batch_id(thread_id));
}

INSTANTIATE_CLASS_CPU_FB(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
