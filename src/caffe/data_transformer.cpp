#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

bool DataTransformer::image_random_crop_enabled() const {
  return this->phase_ == TRAIN && param_.crop_size() > 0;
}

bool DataTransformer::image_center_crop_enabled() const {
  return this->phase_ == TEST && param_.crop_size() > 0;
}

DataTransformer::DataTransformer(const TransformationParameter& param, Phase phase)
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

void DataTransformer::image_random_resize(int new_size, const cv::Mat& src, cv::Mat& dst) {
  const int img_height = src.rows;
  const int img_width = src.cols;
  const float scale = img_width >= img_height ?
                      static_cast<float>(new_size) / static_cast<float>(img_height) :
                      static_cast<float>(new_size) / static_cast<float>(img_width);
  float new_fheight = scale * static_cast<float>(img_height);
  float new_fwidth = scale * static_cast<float>(img_width);
  const float rel_ar_lower = param_.rand_rel_ar_lower();
  const float rel_ar_upper = param_.rand_rel_ar_upper();
  CHECK_GE(rel_ar_lower, 0) << "rand_rel_ar_lower parameter must be non-negative";
  CHECK_GE(rel_ar_upper, 0) << "rand_rel_ar_upper parameter must be non-negative";
  CHECK_GE(rel_ar_upper, rel_ar_lower) << "rand_rel_ar_upper can't be less than rand_rel_ar_lower";

  const float abs_ar_lower = param_.rand_abs_ar_lower();
  const float abs_ar_upper = param_.rand_abs_ar_upper();
  CHECK_GT(abs_ar_lower, 0) << "rand_abs_ar_lower parameter must be positive";
  CHECK_GT(abs_ar_upper, 0) << "rand_abs_ar_upper parameter must be positive";
  CHECK_GE(abs_ar_upper, abs_ar_lower) << "rand_abs_ar_upper can't be less than rand_abs_ar_lower";

  float rel_ratio = new_fwidth >= new_fheight ?
                    new_fwidth / new_fheight : new_fheight / new_fwidth;
  const float rel_ar = Rand(rel_ar_lower, rel_ar_upper);
  rel_ratio = 1.F + (rel_ratio - 1.F) * rel_ar;
  if (new_fwidth > new_fheight) {
    new_fwidth = new_fheight * rel_ratio;
  } else if (new_fwidth < new_fheight) {
    new_fheight = new_fwidth * rel_ratio;
  } else if (Rand(2) > 0) {
    new_fwidth = new_fheight * rel_ratio;
  } else {
    new_fheight = new_fwidth * rel_ratio;
  }

  const float abs_ar = Rand(abs_ar_lower, abs_ar_upper);
  if (abs_ar < 1.F) {
    if (new_fwidth > new_fheight) {
      new_fheight = new_fwidth * abs_ar;
    } else {
      new_fwidth = new_fheight * abs_ar;
    }
  } else if (abs_ar > 1.F) {
    if (new_fwidth > new_fheight) {
      new_fwidth = new_fheight * abs_ar;
    } else {
      new_fheight = new_fwidth * abs_ar;
    }
  }
  int new_height = std::max((int)param_.crop_size(), static_cast<int>(std::round(new_fheight)));
  int new_width = std::max((int)param_.crop_size(), static_cast<int>(std::round(new_fwidth)));

  if (new_height == img_height && new_width == img_width) {
    dst = src;
  } else {
    cv::resize(
        src, dst,
        cv::Size(new_width, new_height),
        0., 0.,
        new_height <= img_height && new_width <= img_width ?
        (int)param_.interpolation_algo_down() : (int)param_.interpolation_algo_up());
  }
}

bool DataTransformer::image_random_resize_enabled() const {
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

void DataTransformer::image_random_crop(int crop_w, int crop_h, cv::Mat& img) {
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

void DataTransformer::image_center_crop(int crop_w, int crop_h, cv::Mat& img) {
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

void DataTransformer::apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst) {
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
      mean_mat_.convertTo(mean_mat_, CVFC<float>(ch), scale);
    }
  } else if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == ch)
        << "Specify either 1 mean_value or as many as channels: " << ch;
    if (src.rows != mean_mat_.rows || src.cols != mean_mat_.cols) {
      if (ch == 3) {
        const int i1 = mean_values_.size() == 1 ? 0 : 1;
        const int i2 = mean_values_.size() == 1 ? 0 : 2;
        cv::Scalar_<float> s(scale * mean_values_[0],
            scale * mean_values_[i1], scale * mean_values_[i2]);
        mean_mat_ = cv::Mat(src.rows, src.cols, CVFC<float>(3), s);
      } else {
        cv::Scalar_<float> s(scale * mean_values_[0]);
        mean_mat_ = cv::Mat(src.rows, src.cols, CVFC<float>(1), s);
      }
    }
  }

  const bool do_mirror = param_.mirror() && Rand(2) > 0;
  src.convertTo(tmp_, CVFC<float>(ch), scale);  // scale & convert
  dst = tmp_;
  if (has_mean_file || has_mean_values) {
    cv::subtract(tmp_, mean_mat_, dst, cv::noArray(), CVFC<float>(ch));  // src-mean -> dst
    if (do_mirror) {
      tmp_ = dst;
    }
  }
  if (do_mirror) {
    cv::flip(tmp_, dst, 1);  // y axis flip
  }
}

void DataTransformer::InitRand() {
  // Use random_seed setting for deterministic transformations
  const uint64_t random_seed = param_.random_seed() >= 0 ?
      static_cast<uint64_t>(param_.random_seed()) : Caffe::next_seed();
  rng_.reset(new Caffe::RNG(random_seed));
}

unsigned int DataTransformer::Rand() const {
  CHECK(rng_);
  caffe::rng_t *rng = static_cast<caffe::rng_t*>(rng_->generator());
  // this doesn't actually produce a uniform distribution
  return (*rng)();
}

float DataTransformer::Rand(float a, float b) const {
  if (a == b) {
    return a;
  }
  double lo = a < b ? a : b;
  double up = a < b ? b : a;
  double r = static_cast<double>(Rand());
  return static_cast<float>(lo + (up - lo) * r / UM);
}

// tests only, TODO: clean
void DataTransformer::VariableSizedTransforms(Datum* datum) {
  cv::Mat img1, img2;
  const int color_mode = param_.force_color() ? 1 : (param_.force_gray() ? -1 : 0);
  if (datum->encoded()) {
    DecodeDatumToCVMat(*datum, color_mode, img1, false);
  } else {
    DatumToCVMat(*datum, img1, false);
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

void DataTransformer::Fill3Randoms(unsigned int *rand) const {
  rand[0] = rand[1] = rand[2] = 0;
  if (param_.mirror()) {
    rand[0] = Rand() + 1;
  }
  if (phase_ == TRAIN && param_.crop_size()) {
    rand[1] = Rand() + 1;
    rand[2] = Rand() + 1;
  }
}

}  // namespace caffe
