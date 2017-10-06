#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param, Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0)
        << "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    TBlobDataToCVMat(data_mean_, mean_mat_orig_);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(!param_.has_mean_file()) << "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.emplace_back(param_.mean_value(c));
    }
  }
  InitRand();
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::Transform(const Datum* datum, Dtype* buf, size_t buf_len) {
  vector<int> shape;
  const bool shape_only = buf == nullptr;
  CHECK(!(param_.force_color() && param_.force_gray()))
      << "cannot set both force_color and force_gray";
  const int color_mode = param_.force_color() ? 1 : (param_.force_gray() ? -1 : 0);
  cv::Mat img;
  bool v1_path = false;
  if (datum->encoded()) {
    shape = DecodeDatumToCVMat(*datum, color_mode, img, shape_only, false);
  } else {
    if (image_random_resize_enabled() || buf == nullptr || buf_len == 0UL) {
      shape = DatumToCVMat<Dtype>(*datum, img, shape_only);
    } else {
      // here we can use fast V1 path
      TransformV1(*datum, buf, buf_len);
      shape = vector<int>{1, datum->channels(), datum->height(), datum->width()};
      v1_path = true;
    }
  }
  if (param_.crop_size() > 0) {
    shape[2] = param_.crop_size();
    shape[3] = param_.crop_size();
  }
  if (!shape_only && !v1_path) {
    CHECK_NOTNULL(img.data);
    Transform(img, buf, buf_len);
  }
  return shape;
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformV1(const Datum& datum, Dtype* buf, size_t buf_len) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const float scale = param_.scale();
  const bool do_mirror = param_.mirror() && (Rand() % 2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  const float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
        << "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand() % (datum_height - crop_size + 1);
      w_off = Rand() % (datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  int top_index, data_index, ch, cdho;
  const int m = do_mirror ? -1 : 1;

  if (has_uint8) {
    Dtype datum_element, mnv;

    if (scale == 1.F) {
      for (int c = 0; c < datum_channels; ++c) {
        cdho = c * datum_height + h_off;
        ch = c * height;
        mnv = has_mean_values && !has_mean_file ? mean_values_[c] : 0.F;
        for (int h = 0; h < height; ++h) {
          top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
          data_index = (cdho + h) * datum_width + w_off;
          for (int w = 0; w < width; ++w) {
            datum_element = static_cast<unsigned char>(data[data_index]);
            if (has_mean_file) {
              buf[top_index] = datum_element - mean[data_index];
            } else {
              if (has_mean_values) {
                buf[top_index] = datum_element - mnv;
              } else {
                buf[top_index] = datum_element;
              }
            }
            ++data_index;
            top_index += m;
          }
        }
      }
    } else {
      for (int c = 0; c < datum_channels; ++c) {
        cdho = c * datum_height + h_off;
        ch = c * height;
        mnv = has_mean_values && !has_mean_file ? mean_values_[c] : 0.F;
        for (int h = 0; h < height; ++h) {
          top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
          data_index = (cdho + h) * datum_width + w_off;
          for (int w = 0; w < width; ++w) {
            datum_element = static_cast<unsigned char>(data[data_index]);
            if (has_mean_file) {
              buf[top_index] = (datum_element - mean[data_index]) * scale;
            } else {
              if (has_mean_values) {
                buf[top_index] = (datum_element - mnv) * scale;
              } else {
                buf[top_index] = datum_element * scale;
              }
            }
            ++data_index;
            top_index += m;
          }
        }
      }
    }
  } else {
    Dtype datum_element;
    for (int c = 0; c < datum_channels; ++c) {
      cdho = c * datum_height + h_off;
      ch = c * height;
      for (int h = 0; h < height; ++h) {
        top_index = do_mirror ? (ch + h + 1) * width - 1 : (ch + h) * width;
        data_index = (cdho + h) * datum_width + w_off;
        for (int w = 0; w < width; ++w) {
          datum_element = datum.float_data(data_index);
          if (has_mean_file) {
            buf[top_index] = (datum_element - mean[data_index]) * scale;
          } else {
            if (has_mean_values) {
              buf[top_index] = (datum_element - mean_values_[c]) * scale;
            } else {
              buf[top_index] = datum_element * scale;
            }
          }
          ++data_index;
          top_index += m;
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& src, Dtype* buf, size_t buf_len) {
  cv::Mat tmp, dst;
  if (image_random_resize_enabled()) {
    int lower_sz = param_.img_rand_resize_lower();
    int upper_sz = param_.img_rand_resize_upper();
    CHECK_GT(lower_sz, 0) << "random resize lower bound parameter must be positive";
    CHECK_GT(upper_sz, 0) << "random resize upper bound parameter must be positive";
    CHECK_GE(upper_sz, lower_sz) << "random resize upper bound can't be less than lower";

    const int new_size = lower_sz + Rand(upper_sz - lower_sz + 1);
    if (new_size > 0) {
      image_random_resize(new_size, src, tmp);
    } else {
      tmp = src;
    }
  } else {
    tmp = src;
  }
  if (image_random_crop_enabled()) {
    image_random_crop(param_.crop_size(), param_.crop_size(), tmp);  // TODO
  } else if (image_center_crop_enabled()) {
    image_center_crop(param_.crop_size(), param_.crop_size(), tmp);
  }
  apply_mean_scale_mirror(tmp, dst);
  FloatCVMatToBuf<Dtype>(dst, buf_len, buf);
}

template<typename Dtype>
bool DataTransformer<Dtype>::image_random_resize_enabled() const {
  const int resize_lower = param_.img_rand_resize_lower();
  const int resize_upper = param_.img_rand_resize_upper();
  if (resize_lower == 0 && resize_upper == 0) {
    return false;
  } else if (resize_lower != 0 && resize_upper != 0) {
    return true;
  }
  LOG(FATAL)
      << "random resize 'lower' and 'upper' parameters must either "
          "both be zero or both be nonzero";
  return false;
}

template<typename Dtype>
void DataTransformer<Dtype>::image_random_resize(int new_size, const cv::Mat& src, cv::Mat& dst) {
  const int img_height = src.rows;
  const int img_width = src.cols;
  const float scale = img_width >= img_height ?
                      static_cast<float>(new_size) / static_cast<float>(img_height) :
                      static_cast<float>(new_size) / static_cast<float>(img_width);
  const int resize_height = static_cast<int>(std::round(scale * static_cast<float>(img_height)));
  const int resize_width = static_cast<int>(std::round(scale * static_cast<float>(img_width)));

  if (resize_height < img_height || resize_width < img_width) {
    CHECK_LE(scale, 1.0F);
    CHECK_LE(resize_height, img_height) << "cannot downsample width without downsampling height";
    CHECK_LE(resize_width, img_width) << "cannot downsample height without downsampling width";
    cv::resize(
        src, dst,
        cv::Size(resize_width, resize_height),
        0., 0.,
        cv::INTER_NEAREST);
  } else if (resize_height > img_height || resize_width > img_width) {
    // Upsample with cubic interpolation.
    CHECK_GE(scale, 1.0F);
    CHECK_GE(resize_height, img_height) << "cannot upsample width without upsampling height";
    CHECK_GE(resize_width, img_width) << "cannot upsample height without upsampling width";
    cv::resize(
        src, dst,
        cv::Size(resize_width, resize_height),
        0., 0.,
        cv::INTER_CUBIC);
  } else if (resize_height == img_height && resize_width == img_width) {
    dst = src;
  } else {
    LOG(FATAL)
        << "unreachable random resize shape: ("
        << img_width << ", " << img_height << ") => ("
        << resize_width << ", " << resize_height << ")";
  }
}

template<typename Dtype>
bool DataTransformer<Dtype>::image_random_crop_enabled() const {
  return this->phase_ == TRAIN && param_.crop_size() > 0;
}

template<typename Dtype>
void DataTransformer<Dtype>::image_random_crop(int crop_w, int crop_h, cv::Mat& img) {
  CHECK_GT(crop_w, 0) << "crop_w parameter must be positive";
  CHECK_GT(crop_h, 0) << "crop_h parameter must be positive";
  const int img_width = img.cols;
  const int img_height = img.rows;
  CHECK_GE(img_width, crop_w) << "crop_w must be at least as large as the image width";
  CHECK_GE(img_height, crop_h) << "crop_h must be at least as large as the image height";
  if (img_width == crop_w && img_height == crop_h) {
    return;
  }
  int crop_offset_h = img_height == crop_h ? 0 : Rand(img_height - crop_h + 1);
  int crop_offset_w = img_width == crop_w ? 0 : Rand(img_width - crop_w + 1);
  cv::Rect roi(crop_offset_w, crop_offset_h, crop_w, crop_h);
  img = img(roi).clone();
}

template<typename Dtype>
bool DataTransformer<Dtype>::image_center_crop_enabled() const {
  return this->phase_ == TEST && param_.crop_size() > 0;
}

template<typename Dtype>
void DataTransformer<Dtype>::image_center_crop(int crop_w, int crop_h, cv::Mat& img) {
  CHECK_GT(crop_w, 0) << "center crop_w parameter must be positive";
  CHECK_GT(crop_h, 0) << "center crop_h parameter must be positive";
  const int img_width = img.cols;
  const int img_height = img.rows;
  CHECK_GE(img_width, crop_w) << "crop w parameter must be at least as large as the image width";
  CHECK_GE(img_height, crop_h)<< "crop_h parameter must be at least as large as the image height";
  if (img_width == crop_w && img_height == crop_h) {
    return;
  }
  const int crop_offset_w = (img_width - crop_w) / 2;
  const int crop_offset_h = (img_height - crop_h) / 2;
  cv::Rect roi(crop_offset_w, crop_offset_h, crop_w, crop_h);
  img = img(roi).clone();
}

template<typename Dtype>
void DataTransformer<Dtype>::apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst) {
  const float scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = !mean_values_.empty();
  const int ch = src.channels();
  if (has_mean_file) {
    CHECK_EQ(ch, mean_mat_orig_.channels());
    if (src.rows != mean_mat_.rows || src.cols != mean_mat_.cols) {
      mean_mat_ = mean_mat_orig_;
      image_center_crop(src.cols, src.rows, mean_mat_);
      // scale & convert in place
      mean_mat_.convertTo(mean_mat_, CVFC<Dtype>(ch), scale);
    }
  } else if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == ch)
        << "Specify either 1 mean_value or as many as channels: " << ch;
    if (src.rows != mean_mat_.rows || src.cols != mean_mat_.cols) {
      if (ch == 3) {
        const int i1 = mean_values_.size() == 1 ? 0 : 1;
        const int i2 = mean_values_.size() == 1 ? 0 : 2;
        cv::Scalar_<Dtype> s(scale * mean_values_[0],
            scale * mean_values_[i1], scale * mean_values_[i2]);
        mean_mat_ = cv::Mat(src.rows, src.cols, CVFC<Dtype>(3), s);
      } else {
        cv::Scalar_<Dtype> s(scale * mean_values_[0]);
        mean_mat_ = cv::Mat(src.rows, src.cols, CVFC<Dtype>(1), s);
      }
    }
  }

  const bool do_mirror = param_.mirror() && Rand(2) > 0;
  src.convertTo(tmp_, CVFC<Dtype>(ch), scale);  // scale & convert
  dst = tmp_;
  if (has_mean_file || has_mean_values) {
    cv::subtract(tmp_, mean_mat_, dst, cv::noArray(), CVFC<Dtype>(ch));  // src-mean -> dst
    if (do_mirror) {
      tmp_ = dst;
    }
  }
  if (do_mirror) {
    cv::flip(tmp_, dst, 1);  // y axis flip
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat>& mat_vector,
    TBlob<Dtype> *transformed_blob) {
  const size_t mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) << "The size of mat_vector must be equals to transformed_blob->num()";
  cv::Mat dst;
  size_t buf_len = transformed_blob->offset(1);
  for (size_t item_id = 0; item_id < mat_num; ++item_id) {
    size_t offset = transformed_blob->offset(item_id);
    apply_mean_scale_mirror(mat_vector[item_id], dst);
    FloatCVMatToBuf<Dtype>(dst, buf_len, transformed_blob->mutable_cpu_data() + offset);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum>& datum_vector,
    TBlob<Dtype> *transformed_blob) {
  const size_t datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num)
    << "The size of datum_vector must be not greater than transformed_blob->num()";
  cv::Mat dst;
  size_t buf_len = transformed_blob->offset(1);
  for (size_t item_id = 0; item_id < datum_num; ++item_id) {
    size_t offset = transformed_blob->offset(item_id);
    DatumToCVMat<Dtype>(datum_vector[item_id], dst, false);
    FloatCVMatToBuf<Dtype>(dst, buf_len, transformed_blob->mutable_cpu_data() + offset);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& img, TBlob<Dtype> *transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = img.channels();
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);
  // TODO
  if (crop_size > 0) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  }
  Transform(img, transformed_blob->mutable_cpu_data(), transformed_blob->count());
}

// tests only, TODO: clean
template<typename Dtype>
void DataTransformer<Dtype>::Transform(Datum& datum, TBlob<Dtype>* transformed_blob) {
  cv::Mat img;
  DatumToCVMat<Dtype>(datum, img, false);
  Transform(img, transformed_blob);
}

// tests only, TODO: clean
template<typename Dtype>
void DataTransformer<Dtype>::VariableSizedTransforms(Datum* datum) {
  cv::Mat img1, img2;
  const int color_mode = param_.force_color() ? 1 : (param_.force_gray() ? -1 : 0);
  if (datum->encoded()) {
    DecodeDatumToCVMat(*datum, color_mode, img1, false);
  } else {
    DatumToCVMat<Dtype>(*datum, img1, false);
  }
  if (image_random_resize_enabled()) {
    const int lower_sz = param_.img_rand_resize_lower();
    const int upper_sz = param_.img_rand_resize_upper();
    const int new_size = lower_sz + Rand(upper_sz - lower_sz + 1);
    image_random_resize(new_size, img1, img2);
  } else {
    img2 = img1;
  }
  if (image_random_crop_enabled()) {
    image_random_crop(param_.crop_size(), param_.crop_size(), img2);
  }
  if (image_center_crop_enabled()) {
    image_center_crop(param_.crop_size(), param_.crop_size(), img2);
  }
  CVMatToDatum(img2, *datum);
}

template<typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  // Use random_seed setting for deterministic transformations
  const uint64_t random_seed = param_.random_seed() >= 0 ?
      static_cast<uint64_t>(param_.random_seed()) : Caffe::next_seed();
  rng_.reset(new Caffe::RNG(random_seed));
}

template<typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() const {
  CHECK(rng_);
  caffe::rng_t *rng = static_cast<caffe::rng_t*>(rng_->generator());
  // this doesn't actually produce a uniform distribution
  return static_cast<unsigned int>((*rng)());
}

INSTANTIATE_CLASS_CPU(DataTransformer);

}  // namespace caffe
