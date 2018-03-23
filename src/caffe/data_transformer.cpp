#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template<typename Dtype>
bool DataTransformer<Dtype>::image_random_crop_enabled() const {
  return this->phase_ == TRAIN && param_.crop_size() > 0;
}

template<typename Dtype>
bool DataTransformer<Dtype>::image_center_crop_enabled() const {
  return this->phase_ == TEST && param_.crop_size() > 0;
}

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param, Phase phase)
    : param_(param), phase_(phase),
      rand_resize_ratio_lower_(param_.rand_resize_ratio_lower()),
      rand_resize_ratio_upper_(param_.rand_resize_ratio_upper()),
      vertical_stretch_lower_(param_.vertical_stretch_lower()),
      vertical_stretch_upper_(param_.vertical_stretch_upper()),
      horizontal_stretch_lower_(param_.horizontal_stretch_lower()),
      horizontal_stretch_upper_(param_.horizontal_stretch_upper()),
      allow_upscale_(param_.allow_upscale()) {
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
void DataTransformer<Dtype>::image_random_resize(const cv::Mat& src, cv::Mat& dst) const {
  if (!image_random_resize_enabled()) {
    dst = src;
    return;
  }
  const int img_height = src.rows;
  const int img_width = src.cols;
  int new_size = std::min(img_height, img_width);

  int lower_sz = param_.img_rand_resize_lower();
  int upper_sz = param_.img_rand_resize_upper();
  if (lower_sz > 0 && upper_sz > 0) {
    CHECK_GE(upper_sz, lower_sz) << "random resize upper bound can't be less than lower";
    new_size = lower_sz + Rand(upper_sz - lower_sz + 1);
  }
  const float scale = img_width >= img_height ?
                      static_cast<float>(new_size) / static_cast<float>(img_height) :
                      static_cast<float>(new_size) / static_cast<float>(img_width);
  float new_fheight = scale * static_cast<float>(img_height);
  float new_fwidth = scale * static_cast<float>(img_width);

  if (rand_resize_ratio_lower_ >= 1.F && rand_resize_ratio_upper_ > 1.F) {
    const float ratio = Rand(rand_resize_ratio_lower_, rand_resize_ratio_upper_);
    if (new_fwidth > new_fheight) {
      new_fwidth = new_fheight * ratio;
    } else if (new_fwidth < new_fheight) {
      new_fheight = new_fwidth * ratio;
    } else if (Rand(2) > 0) {
      new_fwidth = new_fheight * ratio;
    } else {
      new_fheight = new_fwidth * ratio;
    }
  }

  if (vertical_stretch_lower_ > 0.F && vertical_stretch_upper_ > 0.F) {
    const float vertical_stretch = Rand(vertical_stretch_lower_, vertical_stretch_upper_);
    new_fheight *= vertical_stretch;
  }
  if (horizontal_stretch_lower_ > 0.F && horizontal_stretch_upper_ > 0.F) {
    const float horizontal_stretch = Rand(horizontal_stretch_lower_, horizontal_stretch_upper_);
    new_fwidth *= horizontal_stretch;
  }
  int new_height = std::max((int)param_.crop_size(), static_cast<int>(std::lround(new_fheight)));
  int new_width = std::max((int)param_.crop_size(), static_cast<int>(std::lround(new_fwidth)));
  if (!allow_upscale_) {
    if (new_height > img_height) {
      new_height = img_height;
    }
    if (new_width > img_width) {
      new_width = img_width;
    }
  }

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

template<typename Dtype>
bool DataTransformer<Dtype>::image_random_resize_enabled() const {
  const int resize_lower = param_.img_rand_resize_lower();
  const int resize_upper = param_.img_rand_resize_upper();
  const bool use_rand_resize =
      this->param_.has_rand_resize_ratio_lower() ||
      this->param_.has_rand_resize_ratio_upper() ||
      this->param_.has_vertical_stretch_lower() ||
      this->param_.has_vertical_stretch_upper() ||
      this->param_.has_horizontal_stretch_lower() ||
      this->param_.has_horizontal_stretch_upper();
  return resize_lower != 0 || resize_upper != 0 || use_rand_resize;
}

template<typename Dtype>
void DataTransformer<Dtype>::image_random_crop(int crop_w, int crop_h, cv::Mat& img) const {
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
void DataTransformer<Dtype>::image_center_crop(int crop_w, int crop_h, cv::Mat& img) const {
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
void DataTransformer<Dtype>::apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst) const {
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
  return (*rng)();
}

template<typename Dtype>
float DataTransformer<Dtype>::Rand(float a, float b) const {
  if (a == b) {
    return a;
  }
  double lo = a < b ? a : b;
  double up = a < b ? b : a;
  double r = static_cast<double>(Rand());
  return static_cast<float>(lo + (up - lo) * r / UM);
}

template<typename Dtype>
void DataTransformer<Dtype>::Copy(const cv::Mat& cv_img, Dtype *data) {
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  int top_index;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      const uchar *ptr = cv_img.ptr<uchar>(h);
      for (int w = 0; w < width; ++w) {
        int img_index = w * channels + c;
        top_index = (c * height + h) * width + w;
        data[top_index] = static_cast<Dtype>(ptr[img_index]);
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    Copy(cv_img, data);
    out_sizeof_element = sizeof(Dtype);
    return;
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "Force_color and force_gray are for encoded datum only";
    }
  }

  const string& datum_data = datum.data();
  const int N = datum.channels() * datum.height() * datum.width();
  const void* src_ptr;
  if (datum_data.size() > 0) {
    CHECK_LE(sizeof(uint8_t), sizeof(Dtype));
    CHECK_EQ(N, datum_data.size());
    out_sizeof_element = sizeof(uint8_t);
    src_ptr = &datum_data.front();
  } else {
    CHECK_LE(sizeof(float), sizeof(Dtype));
    out_sizeof_element = sizeof(float);
    src_ptr = &datum.float_data().Get(0);
  }
  cudaStream_t stream = Caffe::thread_stream();
  CUDA_CHECK(cudaMemcpyAsync(data, src_ptr, N * out_sizeof_element,
      cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Dtype>
void DataTransformer<Dtype>::Fill3Randoms(unsigned int *rand) const {
  rand[0] = rand[1] = rand[2] = 0;
  if (param_.mirror()) {
    rand[0] = Rand() + 1;
  }
  if (phase_ == TRAIN && param_.crop_size()) {
    rand[1] = Rand() + 1;
    rand[2] = Rand() + 1;
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::Transform(const Datum* datum, Dtype* buf, size_t buf_len,
    Packing& out_packing, bool repack) {
  vector<int> shape;
  const bool shape_only = buf == nullptr;
  CHECK(!(param_.force_color() && param_.force_gray()))
  << "cannot set both force_color and force_gray";
  const int color_mode = param_.force_color() ? 1 : (param_.force_gray() ? -1 : 0);
  cv::Mat img;
  bool v1_path = false;
  if (datum->encoded()) {
    shape = DecodeDatumToCVMat(*datum, color_mode, img, shape_only, false);
    out_packing = NHWC;
  } else {
    if (image_random_resize_enabled() || buf == nullptr || buf_len == 0UL) {
      shape = DatumToCVMat(*datum, img, shape_only);
      out_packing = NHWC;
    } else {
      // here we can use fast V1 path
      TransformV1(*datum, buf, buf_len);
      shape = vector<int>{1, datum->channels(), datum->height(), datum->width()};
      v1_path = true;
      out_packing = NCHW;
    }
  }
  if (param_.crop_size() > 0) {
    shape[2] = param_.crop_size();
    shape[3] = param_.crop_size();
  }
  if (!shape_only && !v1_path) {
    CHECK_NOTNULL(img.data);
    Transform(img, buf, buf_len, repack);
    out_packing = NHWC;
  }
  return shape;
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& src, Dtype* buf, size_t buf_len,
    bool repack) const {
  cv::Mat tmp, dst;

  image_random_resize(src, tmp);

  const int crop_w = param_.crop_w() > 0 ? param_.crop_w() : param_.crop_size();
  const int crop_h = param_.crop_h() > 0 ? param_.crop_h() : param_.crop_size();
  if (image_random_crop_enabled()) {
    image_random_crop(crop_w, crop_h, tmp);
  } else if (image_center_crop_enabled()) {
    image_center_crop(crop_w, crop_h, tmp);
  }
  apply_mean_scale_mirror(tmp, dst);
  FloatCVMatToBuf<Dtype>(dst, buf_len, buf, repack);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat>& mat_vector,
    TBlob<Dtype>* transformed_blob) const {
  const size_t mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) << "The size of mat_vector must be equals to transformed_blob->num()";
  cv::Mat dst;
  size_t buf_len = transformed_blob->offset(1);
  for (size_t item_id = 0; item_id < mat_num; ++item_id) {
    size_t offset = transformed_blob->offset(item_id);
    apply_mean_scale_mirror(mat_vector[item_id], dst);
    FloatCVMatToBuf<Dtype>(dst, buf_len, transformed_blob->mutable_cpu_data(false) + offset);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum>& datum_vector,
    TBlob<Dtype> *transformed_blob) const {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num)
    << "The size of datum_vector must be no greater than transformed_blob->num()";
  TBlob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       TBlob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob, crop_bbox, do_mirror);
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data, crop_bbox, do_mirror);
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
    bool* do_mirror) {
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  NormalizedBBox crop_bbox;
  Transform(datum, transformed_blob, &crop_bbox, do_mirror);

  // Transform annotation.
  const bool do_resize = true;
  TransformAnnotation(anno_datum, do_resize, crop_bbox, *do_mirror,
                      transformed_anno_group_all);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_group_all,
            &do_mirror);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec, bool* do_mirror) {
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all,
            do_mirror);
  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(
    const AnnotatedDatum& anno_datum, TBlob<Dtype>* transformed_blob,
    vector<AnnotationGroup>* transformed_anno_vec) {
  bool do_mirror;
  Transform(anno_datum, transformed_blob, transformed_anno_vec, &do_mirror);
}

// Transform and return the transformation information.
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data,
               NormalizedBBox* crop_bbox, bool* do_mirror) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
        "Specify either 1 mean_value or as many as channels: " << datum_channels;
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
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (*do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
              (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
                (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::TransformAnnotation(
    const AnnotatedDatum& anno_datum, const bool do_resize,
    const NormalizedBBox& crop_bbox, const bool do_mirror,
    RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
    // Go through each AnnotationGroup.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.has_resize_param()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
                                   &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.has_resize_param()) {
            ExtrapolateBBox(param_.resize_param(), img_height, img_width,
                crop_bbox, transformed_bbox);
          }
        }
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
  } else {
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const Datum& datum,
                                       const NormalizedBBox& bbox,
                                       Datum* crop_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Crop the image.
    cv::Mat crop_img;
    CropImage(cv_img, bbox, &crop_img);
    // Save the image into datum.
    EncodeCVMatToDatum(crop_img, "jpg", crop_datum);
    crop_datum->set_label(datum.label());
    return;
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
  const int w_off = static_cast<int>(scaled_bbox.xmin());
  const int h_off = static_cast<int>(scaled_bbox.ymin());
  const int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  const int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());

  // Crop the image using bbox.
  crop_datum->set_channels(datum_channels);
  crop_datum->set_height(height);
  crop_datum->set_width(width);
  crop_datum->set_label(datum.label());
  crop_datum->clear_data();
  crop_datum->clear_float_data();
  crop_datum->set_encoded(false);
  const int crop_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(crop_datum_size, ' ');
  for (int h = h_off; h < h_off + height; ++h) {
    for (int w = w_off; w < w_off + width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        int crop_datum_index = (c * height + h - h_off) * width + w - w_off;
        buffer[crop_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  crop_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const Datum& datum,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         Datum* expand_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Expand the image.
    cv::Mat expand_img;
    ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
    // Save the image into datum.
    EncodeCVMatToDatum(expand_img, "jpg", expand_datum);
    expand_datum->set_label(datum.label());
    return;
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  int height = static_cast<int>(datum_height * expand_ratio);
  int width = static_cast<int>(datum_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - datum_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - datum_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/datum_width);
  expand_bbox->set_ymin(-h_off/datum_height);
  expand_bbox->set_xmax((width - w_off)/datum_width);
  expand_bbox->set_ymax((height - h_off)/datum_height);

  // Crop the image using bbox.
  expand_datum->set_channels(datum_channels);
  expand_datum->set_height(height);
  expand_datum->set_width(width);
  expand_datum->set_label(datum.label());
  expand_datum->clear_data();
  expand_datum->clear_float_data();
  expand_datum->set_encoded(false);
  const int expand_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(expand_datum_size, ' ');
  for (int h = h_off; h < h_off + datum_height; ++h) {
    for (int w = w_off; w < w_off + datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index =
            (c * datum_height + h - h_off) * datum_width + w - w_off;
        int expand_datum_index = (c * height + h) * width + w;
        buffer[expand_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  expand_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const AnnotatedDatum& anno_datum,
                                         AnnotatedDatum* expanded_anno_datum) {
  if (!param_.has_expand_param()) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const ExpansionParameter& expand_param = param_.expand_param();
  const float expand_prob = expand_param.prob();
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const float max_expand_ratio = expand_param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
  // Expand the datum.
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
              expanded_anno_datum->mutable_datum());
  expanded_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
                      expanded_anno_datum->mutable_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::DistortImage(const Datum& datum,
                                          Datum* distort_datum) {
  if (!param_.has_distort_param()) {
    distort_datum->CopyFrom(datum);
    return;
  }
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Distort the image.
    cv::Mat distort_img = ApplyDistort(cv_img, param_.distort_param());
    // Save the image into datum.
    EncodeCVMatToDatum(distort_img, "jpg", distort_datum);
    distort_datum->set_label(datum.label());
    return;
  } else {
    LOG(ERROR) << "Only support encoded datum now";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       TBlob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       bool* do_mirror) {
  // Check dimensions.
  const int img_channels = cv_img.channels();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_GT(img_channels, 0);
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  *do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  cv::Mat cv_resized_image, cv_noised_image, cv_cropped_image;
  if (param_.has_resize_param()) {
    cv_resized_image = ApplyResize(cv_img, param_.resize_param());
  } else {
    cv_resized_image = cv_img;
  }
  if (param_.has_noise_param()) {
    cv_noised_image = ApplyNoise(cv_resized_image, param_.noise_param());
  } else {
    cv_noised_image = cv_resized_image;
  }
  int img_height = cv_noised_image.rows;
  int img_width = cv_noised_image.cols;
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  int h_off = 0;
  int w_off = 0;
  if ((crop_h > 0) && (crop_w > 0)) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    } else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_image = cv_noised_image(roi);
  } else {
    cv_cropped_image = cv_noised_image;
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / img_width);
  crop_bbox->set_ymin(Dtype(h_off) / img_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / img_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / img_height);

  if (has_mean_file) {
    CHECK_EQ(cv_cropped_image.rows, data_mean_.height());
    CHECK_EQ(cv_cropped_image.cols, data_mean_.width());
  }
  CHECK(cv_cropped_image.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_image.ptr<uchar>(h);
    int img_index = 0;
    int h_idx = h;
    for (int w = 0; w < width; ++w) {
      int w_idx = w;
      if (*do_mirror) {
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      for (int c = 0; c < img_channels; ++c) {
        top_index = (c * height + h_idx_real) * width + w_idx_real;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h_idx_real) * img_width
              + w_off + w_idx_real;
          transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Dtype* data, cv::Mat* cv_img,
                                          const int height, const int width,
                                          const int channels) {
  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
        "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));
  for (int h = 0; h < height; ++h) {
    uchar* ptr = orig_img.ptr<uchar>(h);
    int img_idx = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int idx = (c * height + h) * width + w;
        if (has_mean_file) {
          ptr[img_idx++] = static_cast<uchar>(data[idx] / scale + mean[idx]);
        } else {
          if (has_mean_values) {
            ptr[img_idx++] =
                static_cast<uchar>(data[idx] / scale + mean_values_[c]);
          } else {
            ptr[img_idx++] = static_cast<uchar>(data[idx] / scale);
          }
        }
      }
    }
  }

  if (param_.has_resize_param()) {
    *cv_img = ApplyResize(orig_img, param_.resize_param());
  } else {
    *cv_img = orig_img;
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Blob* blob,
                                          vector<cv::Mat>* cv_imgs) {
  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const int num = blob->num();
  CHECK_GE(num, 1);
  const Dtype* image_data = blob->cpu_data<Dtype>();

  for (int i = 0; i < num; ++i) {
    cv::Mat cv_img;
    TransformInv(image_data, &cv_img, height, width, channels);
    cv_imgs->push_back(cv_img);
    image_data += blob->offset(1);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const cv::Mat& img,
                                       const NormalizedBBox& bbox,
                                       cv::Mat* crop_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

  // Crop the image using bbox.
  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  cv::Rect bbox_roi(w_off, h_off, width, height);

  img(bbox_roi).copyTo(*crop_img);
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const cv::Mat& img,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         cv::Mat* expand_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;
  const int img_channels = img.channels();

  // Get the bbox dimension.
  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/img_width);
  expand_bbox->set_ymin(-h_off/img_height);
  expand_bbox->set_xmax((width - w_off)/img_width);
  expand_bbox->set_ymax((height - h_off)/img_height);

  expand_img->create(height, width, img.type());
  expand_img->setTo(cv::Scalar(0));
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    float* mean = data_mean_.mutable_cpu_data();
    for (int h = 0; h < height; ++h) {
      uchar* ptr = expand_img->ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < img_channels; ++c) {
          int blob_index = (c * height + h) * width + w;
          ptr[img_index++] = static_cast<char>(mean[blob_index]);
        }
      }
    }
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    vector<cv::Mat> channels(img_channels);
    cv::split(*expand_img, channels);
    CHECK_EQ(channels.size(), mean_values_.size());
    for (int c = 0; c < img_channels; ++c) {
      channels[c] = mean_values_[c];
    }
    cv::merge(channels, *expand_img);
  }

  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  img.copyTo((*expand_img)(bbox_roi));
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
  }
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), datum_width, datum_height,
                 &datum_width, &datum_height);
  }
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h)? crop_h: datum_height;
  shape[3] = (crop_w)? crop_w: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
                 &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h: img_height;
  shape[3] = (crop_w)? crop_w: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferDatumShape(const Datum& datum) {
  if (datum.encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Infer shape using the cv::image.
    return InferCVMatShape(cv_img);
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  vector<int> datum_shape(4);
  datum_shape[0] = 1;
  datum_shape[1] = datum_channels;
  datum_shape[2] = datum_height;
  datum_shape[3] = datum_width;
  return datum_shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferCVMatShape(const cv::Mat& cv_img) {
  int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = img_height;
  shape[3] = img_width;

  const int crop_size = param_.crop_size();
  int crop_h = param_.crop_h();
  int crop_w = param_.crop_w();
  if (crop_size) {
    crop_h = crop_size;
    crop_w = crop_size;
  }

  // Check dimensions.
  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
        &img_width, &img_height);
  }
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  // Build BlobShape.
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h: img_height;
  shape[3] = (crop_w)? crop_w: img_width;

  return shape;
}


template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const vector<int>& bottom_shape, bool use_gpu) {
  const int crop_size = param_.crop_size();
  CHECK_EQ(bottom_shape.size(), 4);
  CHECK_EQ(bottom_shape[0], 1);
  const int bottom_channels = bottom_shape[1];
  const int bottom_height = bottom_shape[2];
  const int bottom_width = bottom_shape[3];
  // Check dimensions.
  CHECK_GT(bottom_channels, 0);
  CHECK_GE(bottom_height, crop_size);
  CHECK_GE(bottom_width, crop_size);
  // Build BlobShape.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = bottom_channels;
  // if using GPU transform, don't crop
  if (use_gpu) {
    top_shape[2] = bottom_height;
    top_shape[3] = bottom_width;
  } else {
    top_shape[2] = (crop_size) ? crop_size : bottom_height;
    top_shape[3] = (crop_size) ? crop_size : bottom_width;
  }
  return top_shape;
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
    Dtype *transformed_data, const std::array<unsigned int, 3>& rand) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const float scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand[0] % 2);
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
      h_off = rand[1] % (datum_height - crop_size + 1);
      w_off = rand[2] % (datum_width - crop_size + 1);
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
              transformed_data[top_index] = datum_element - mean[data_index];
            } else {
              if (has_mean_values) {
                transformed_data[top_index] = datum_element - mnv;
              } else {
                transformed_data[top_index] = datum_element;
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
              transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
            } else {
              if (has_mean_values) {
                transformed_data[top_index] = (datum_element - mnv) * scale;
              } else {
                transformed_data[top_index] = datum_element * scale;
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
            transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
          } else {
            if (has_mean_values) {
              transformed_data[top_index] = (datum_element - mean_values_[c]) * scale;
            } else {
              transformed_data[top_index] = datum_element * scale;
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
void DataTransformer<Dtype>::Transform(const cv::Mat& img, TBlob<Dtype> *transformed_blob) const {
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
  Transform(img, transformed_blob->mutable_cpu_data(false), transformed_blob->count());
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
    float datum_element, mnv;

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
    float datum_element;
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

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
