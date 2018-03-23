#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/solver.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/rng.hpp"

#define IDL_CACHE_PROGRESS 0.05F

namespace caffe {

static std::mutex idl_mutex_;

template <typename Ftype, typename Btype>
size_t ImageDataLayer<Ftype, Btype>::id(const string& ph, const string& name) {
  std::lock_guard<std::mutex> lock(idl_mutex_);
  static size_t id = 0UL;
  static map<string, size_t> ph_names;
  string ph_name = ph + name;
  auto it = ph_names.find(ph_name);
  if (it != ph_names.end()) {
    return it->second;
  }
  CHECK_LT(id, MAX_IDL_CACHEABLE);
  ph_names.emplace(ph_name, id);
  return id++;
};

template <typename Ftype, typename Btype>
ImageDataLayer<Ftype, Btype>::ImageDataLayer(const LayerParameter& param, size_t solver_rank)
    : BasePrefetchingDataLayer<Ftype, Btype>(param, solver_rank),
      id_(id(Phase_Name(this->phase_), this->name())),
      epoch_count_(0UL) {
  DLOG(INFO) << this->print_current_device() << " ImageDataLayer: " << this
             << " name: " << this->name()
             << " id: " << id_
             << " threads: " << this->threads_num();
}

template <typename Ftype, typename Btype>
ImageDataLayer<Ftype, Btype>::~ImageDataLayer<Ftype, Btype>() {
  if (layer_inititialized_flag_.is_set()) {
    this->StopInternalThread();
  }
}

template <typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::DataLayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const ImageDataParameter& image_data_param = this->layer_param_.image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int short_side = image_data_param.short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color  = image_data_param.is_color();
  const string& root_folder = image_data_param.root_folder();
  vector<std::pair<std::string, int>>& lines = lines_[id_];

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  if (this->rank_ == 0) {
    // Read the file with filenames and labels
    lines.clear();
    const string &source = image_data_param.source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    CHECK(infile.good()) << "File " << source;
    string filename;
    int label;
    while (infile >> filename >> label) {
      lines.emplace_back(std::make_pair(filename, label));
    }
    if (image_data_param.shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
      ShuffleImages();
    }
  }
  LOG(INFO) << this->print_current_device() << " A total of " << lines.size() << " images.";

  size_t skip = 0UL;
  // Check if we would need to randomly skip a few data points
  if (image_data_param.rand_skip()) {
    if (Caffe::gpus().size() > 1) {
      LOG(WARNING) << "Skipping data points is not supported in multiGPU mode";
    } else {
      skip = caffe_rng_rand() % image_data_param.rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points";
      CHECK_GT(lines.size(), skip) << "Not enough points to skip";
    }
  }
  line_ids_.resize(this->threads_num());
  for (size_t i = 0; i < this->threads_num(); ++i) {
    line_ids_[i] = this->rank_ * this->threads_num() + i + skip;
  }

  // Read an image, and use it to initialize the top blob.
  string file_name = lines[line_ids_[0]].first;
  cv::Mat cv_img = next_mat(root_folder, file_name, new_height, new_width, is_color, short_side);
  CHECK(cv_img.data) << "Could not load " << root_folder + file_name;
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = image_data_param.batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  int crop_height = crop;
  int crop_width = crop;
  if (crop <= 0) {
    LOG_FIRST_N(INFO, 1) << "Crop is not set. Using '" << root_folder + file_name
                         << "' as a model, w=" << cv_img.rows << ", h=" << cv_img.cols;
    crop_height = cv_img.rows;
    crop_width = cv_img.cols;
  }
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
  shuffle(lines_[id_].begin(), lines_[id_].end(), prefetch_rng);
}

template<typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::InitializePrefetch() {}

template<typename Ftype, typename Btype>
cv::Mat ImageDataLayer<Ftype, Btype>::next_mat(const string& root_folder, const string& file_name,
                                               int height, int width,
                                               bool is_color, int short_side) {
  if (this->layer_param_.image_data_param().cache()) {
    std::lock_guard<std::mutex> lock(cache_mutex_[id_]);
    if (cache_[id_].size() > 0) {
      auto it = cache_[id_].find(file_name);
      if (it != cache_[id_].end()) {
        return it->second;
      }
    }
  }
  return ReadImageToCVMat(root_folder + file_name, height, width, is_color, short_side);
}

template <typename Ftype, typename Btype>
void ImageDataLayer<Ftype, Btype>::load_batch(Batch* batch, int thread_id, size_t) {
  CHECK(batch->data_->count());
  const ImageDataParameter& image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int short_side = image_data_param.short_side();
  const int crop = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  const bool cache_on = image_data_param.cache();
  const bool shuffle = image_data_param.shuffle();
  const string& root_folder = image_data_param.root_folder();
  unordered_map<std::string, cv::Mat>& cache = cache_[id_];
  vector<std::pair<std::string, int>>& lines = lines_[id_];

  size_t line_id = line_ids_[thread_id];
  const size_t line_bucket = Caffe::gpus().size() * this->threads_num();
  const size_t lines_size = lines.size();
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  string file_name = lines[line_id].first;
  cv::Mat cv_img = next_mat(root_folder, file_name, new_height, new_width, is_color, short_side);

  CHECK(cv_img.data) << "Could not load " << (root_folder + file_name);
  int crop_height = crop;
  int crop_width = crop;
  if (crop <= 0) {
    LOG_FIRST_N(INFO, 1) << "Crop is not set. Using '"
        << (root_folder + file_name)
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
    file_name = lines[line_id].first;
    cv::Mat cv_img = next_mat(root_folder, file_name, new_height, new_width, is_color, short_side);

    if (cv_img.data) {
      int offset = batch->data_->offset(item_id);
#if defined(USE_CUDNN)
      this->bdt(thread_id)->Transform(cv_img, prefetch_data + offset, buf_len, false);
#else
      CHECK_EQ(buf_len, tmp.size());
      this->bdt(thread_id)->Transform(cv_img, prefetch_data + offset, buf_len, false);
      hwc2chw(top_shape[1], top_shape[3], top_shape[2], tmp.data(), prefetch_data + offset);
      packing = NCHW;
#endif
      prefetch_label[item_id] = lines[line_id].second;
    }
    if (cache_on && !cached_[id_]) {
      std::lock_guard<std::mutex> lock(cache_mutex_[id_]);
      if (cv_img.data != nullptr) {
        auto em = cache.emplace(file_name, cv_img);
        if (em.second) {
          ++cached_num_[id_];
        }
      } else {
        ++failed_num_[id_];
      }
      if (cached_num_[id_] + failed_num_[id_] >= lines_size) {
        cached_[id_] = true;
        LOG(INFO) << cache.size() << " objects cached for " << Phase_Name(this->phase_)
                  << " by layer " << this->name();
      } else if ((float) cached_num_[id_] / lines_size >=
          cache_progress_[id_] + IDL_CACHE_PROGRESS) {
        cache_progress_[id_] = (float) cached_num_[id_] / lines_size;
        LOG(INFO) << std::setw(2) << std::setfill(' ') << f_round1(cache_progress_[id_] * 100.F)
                  << "% of objects cached for "
                  << Phase_Name(this->phase_) << " by layer '" << this->name() << "' ("
                  << cached_num_[id_] << "/" << lines_size << ")";
      }
    }

    // go to the next iter
    line_ids_[thread_id] += line_bucket;
    if (line_ids_[thread_id] >= lines_size) {
      while (line_ids_[thread_id] >= lines_size) {
        line_ids_[thread_id] -= lines_size;
      }
      if (thread_id == 0 && this->rank_ == 0) {
        if (this->phase_ == TRAIN) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << this->print_current_device() << " Restarting data prefetching ("
                    << lines_size << ")";
          if (epoch_count_ == 0UL) {
            epoch_count_ += lines_size;
            Caffe::report_epoch_count(epoch_count_);
          }
        }
        if (shuffle) {
          LOG(INFO) << "Shuffling data";
          ShuffleImages();
        }
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
