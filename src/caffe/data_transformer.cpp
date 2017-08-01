#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#endif  // USE_OPENCV

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
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(!param_.has_mean_file())
    << "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

#ifdef USE_OPENCV

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
#endif

template<typename Dtype>
void DataTransformer<Dtype>::Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
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
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "Force_color and force_gray are for encoded datum only";
    }
  }

#ifndef CPU_ONLY
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
  cudaStream_t stream = Caffe::th_stream_aux(Caffe::STREAM_ID_TRANSFORMER);
  CUDA_CHECK(cudaMemcpyAsync(data, src_ptr, N * out_sizeof_element,
      cudaMemcpyHostToDevice, stream));
//  CUDA_CHECK(cudaStreamSynchronize(stream));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
void DataTransformer<Dtype>::CopyPtrEntry(
    shared_ptr<Datum> datum,
    Dtype* transformed_ptr,
    size_t& out_sizeof_element,
    bool output_labels, Dtype *label) {
  if (output_labels) {
    *label = datum->label();
  }
  Copy(*datum, transformed_ptr, out_sizeof_element);
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

#ifdef USE_OPENCV
template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_transforms_enabled() const {
  return var_sized_image_random_resize_enabled() ||
      var_sized_image_random_crop_enabled() ||
      var_sized_image_center_crop_enabled();
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_transforms_shape(
    const vector<int>& orig_shape) const {
  CHECK_EQ(orig_shape.size(), 4);
  // All of the transforms (random resize, random crop, center crop)
  // can be enabled, and they operate sequentially, one after the other.
  vector<int> shape(orig_shape);
  if (var_sized_image_random_resize_enabled()) {
    shape = var_sized_image_random_resize_shape(shape);
  }
  if (var_sized_image_random_crop_enabled()) {
    shape = var_sized_image_random_crop_shape(shape);
  }
  if (var_sized_image_center_crop_enabled()) {
    shape = var_sized_image_center_crop_shape(shape);
  }
  CHECK_NE(shape[2], 0)
      << "variable sized transform has invalid output height; did you forget to crop?";
  CHECK_NE(shape[3], 0)
      << "variable sized transform has invalid output width; did you forget to crop?";
  return shape;
}

template<typename Dtype>
shared_ptr<Datum> DataTransformer<Dtype>::VariableSizedTransforms(shared_ptr<Datum> datum) {
  if (datum->encoded()) {
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      DecodeDatumToCVMat(*datum, param_.force_color(), varsz_orig_img_);
    } else {
      DecodeDatumToCVMatNative(*datum, varsz_orig_img_);
    }
  } else {
    DatumToCVMat(*datum, varsz_orig_img_);
  }
  if (var_sized_image_random_resize_enabled()) {
    varsz_orig_img_ = var_sized_image_random_resize(varsz_orig_img_);
  }
  if (var_sized_image_random_crop_enabled()) {
    varsz_orig_img_ = var_sized_image_random_crop(varsz_orig_img_);
  }
  if (var_sized_image_center_crop_enabled()) {
    varsz_orig_img_ = var_sized_image_center_crop(varsz_orig_img_);
  }
  shared_ptr<Datum> new_datum = make_shared<Datum>();
  CVMatToDatum(varsz_orig_img_, *new_datum);
  if (datum->has_label()) {
    new_datum->set_label(datum->label());
  }
  return new_datum;
}

template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_image_random_resize_enabled() const {
  const int resize_lower = param_.var_sz_img_rand_resize_lower();
  const int resize_upper = param_.var_sz_img_rand_resize_upper();
  if (resize_lower == 0 && resize_upper == 0) {
    return false;
  } else if (resize_lower != 0 && resize_upper != 0) {
    return true;
  }
  LOG(FATAL)
      << "random resize 'lower' and 'upper' parameters must either "
      "both be zero or both be nonzero";
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_image_random_resize_shape(
    const vector<int>& prev_shape) const {
  CHECK(var_sized_image_random_resize_enabled())
      << "var sized transform must be enabled";
  CHECK_EQ(prev_shape.size(), 4)
      << "input shape should always have 4 axes (NCHW)";
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = prev_shape[1];
  // The output of a random resize is itself a variable sized image.
  // By itself a random resize cannot produce an image that is valid input for
  // downstream transformations, and must instead be terminated by a
  // variable-sized crop (either random or center).
  shape[2] = 0;
  shape[3] = 0;
  return shape;
}

template<typename Dtype>
cv::Mat& DataTransformer<Dtype>::var_sized_image_random_resize(cv::Mat& img) {
  const int resize_lower = param_.var_sz_img_rand_resize_lower();
  const int resize_upper = param_.var_sz_img_rand_resize_upper();
  CHECK_GT(resize_lower, 0)
      << "random resize lower bound parameter must be positive";
  CHECK_GT(resize_upper, 0)
      << "random resize lower bound parameter must be positive";
  int resize_size = -1;
  caffe_rng_uniform(
      1,
      static_cast<float>(resize_lower), static_cast<float>(resize_upper),
      &resize_size);
  CHECK_NE(resize_size, -1)
      << "uniform random sampling inexplicably failed";
  const int img_height = img.rows;
  const int img_width = img.cols;
  const double scale = (img_width >= img_height) ?
      ((static_cast<double>(resize_size)) / (static_cast<double>(img_height))) :
      ((static_cast<double>(resize_size)) / (static_cast<double>(img_width)));
  const int resize_height = static_cast<int>(std::round(scale * static_cast<double>(img_height)));
  const int resize_width = static_cast<int>(std::round(scale * static_cast<double>(img_width)));
  if (resize_height < img_height || resize_width < img_width) {
    // Downsample with pixel area relation interpolation.
    CHECK_LE(scale, 1.0);
    CHECK_LE(resize_height, img_height)
        << "cannot downsample width without downsampling height";
    CHECK_LE(resize_width, img_width)
        << "cannot downsample height without downsampling width";
    cv::resize(
        img, varsz_rand_resize_img_,
        cv::Size(resize_width, resize_height),
        0.0, 0.0,
        cv::INTER_AREA);
    return varsz_rand_resize_img_;
  } else if (resize_height > img_height || resize_width > img_width) {
    // Upsample with cubic interpolation.
    CHECK_GE(scale, 1.0);
    CHECK_GE(resize_height, img_height)
        << "cannot upsample width without upsampling height";
    CHECK_GE(resize_width, img_width)
        << "cannot upsample height without upsampling width";
    cv::resize(
        img, varsz_rand_resize_img_,
        cv::Size(resize_width, resize_height),
        0.0, 0.0,
        cv::INTER_CUBIC);
    return varsz_rand_resize_img_;
  } else if (resize_height == img_height && resize_width == img_width) {
    return img;
  }
  LOG(FATAL)
      << "unreachable random resize shape: ("
      << img_width << ", " << img_height << ") => ("
      << resize_width << ", " << resize_height << ")";
}

template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_image_random_crop_enabled() const {
  const int crop_size = param_.var_sz_img_rand_crop();
  return crop_size != 0;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_image_random_crop_shape(
    const vector<int>& prev_shape) const {
  CHECK(var_sized_image_random_crop_enabled())
      << "var sized transform must be enabled";
  const int crop_size = param_.var_sz_img_rand_crop();
  CHECK_EQ(prev_shape.size(), 4)
      << "input shape should always have 4 axes (NCHW)";
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = prev_shape[1];
  shape[2] = crop_size;
  shape[3] = crop_size;
  return shape;
}

template<typename Dtype>
cv::Mat& DataTransformer<Dtype>::var_sized_image_random_crop(const cv::Mat& img) {
  const int crop_size = param_.var_sz_img_rand_crop();
  CHECK_GT(crop_size, 0)
      << "random crop size parameter must be positive";
  const int img_height = img.rows;
  const int img_width = img.cols;
  CHECK_GE(img_height, crop_size)
      << "crop size parameter must be at least as large as the image height";
  CHECK_GE(img_width, crop_size)
      << "crop size parameter must be at least as large as the image width";
  int crop_offset_h = -1;
  int crop_offset_w = -1;
  caffe_rng_uniform(1, 0.0f, static_cast<float>(img_height - crop_size), &crop_offset_h);
  caffe_rng_uniform(1, 0.0f, static_cast<float>(img_width - crop_size), &crop_offset_w);
  CHECK_NE(crop_offset_h, -1)
      << "uniform random sampling inexplicably failed";
  CHECK_NE(crop_offset_w, -1)
      << "uniform random sampling inexplicably failed";
  cv::Rect crop_roi(crop_offset_w, crop_offset_h, crop_size, crop_size);
  varsz_rand_crop_img_ = img(crop_roi);
  return varsz_rand_crop_img_;
}

template<typename Dtype>
bool DataTransformer<Dtype>::var_sized_image_center_crop_enabled() const {
  const int crop_size = param_.var_sz_img_center_crop();
  return crop_size != 0;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::var_sized_image_center_crop_shape(
    const vector<int>& prev_shape) const {
  CHECK(var_sized_image_center_crop_enabled())
      << "var sized transform must be enabled";
  const int crop_size = param_.var_sz_img_center_crop();
  CHECK_EQ(prev_shape.size(), 4)
      << "input shape should always have 4 axes (NCHW)";
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = prev_shape[1];
  shape[2] = crop_size;
  shape[3] = crop_size;
  return shape;
}

template<typename Dtype>
cv::Mat& DataTransformer<Dtype>::var_sized_image_center_crop(const cv::Mat& img) {
  const int crop_size = param_.var_sz_img_center_crop();
  CHECK_GT(crop_size, 0)
      << "center crop size parameter must be positive";
  const int img_height = img.rows;
  const int img_width = img.cols;
  CHECK_GE(img_height, crop_size)
      << "crop size parameter must be at least as large as the image height";
  CHECK_GE(img_width, crop_size)
      << "crop size parameter must be at least as large as the image width";
  const int crop_offset_h = (img_height - crop_size) / 2;
  const int crop_offset_w = (img_width - crop_size) / 2;
  cv::Rect crop_roi(crop_offset_w, crop_offset_h, crop_size, crop_size);
  varsz_center_crop_img_ = img(crop_roi);
  return varsz_center_crop_img_;
}
#endif

#ifndef CPU_ONLY

template<typename Dtype>
void DataTransformer<Dtype>::TransformGPU(const Datum& datum,
    Dtype *transformed_data, const std::array<unsigned int, 3>& rand) {

  unsigned int *randoms =
      reinterpret_cast<unsigned int *>(GPUMemory::thread_pinned_buffer(sizeof(unsigned int) * 3));
  std::memcpy(randoms, &rand.front(), sizeof(unsigned int) * 3);  // NOLINT(caffe/alt_fn)

  vector<int> datum_shape = InferBlobShape(datum, 1);
  TBlob<Dtype> original_data;
  original_data.Reshape(datum_shape);

  Dtype *original_data_gpu_ptr;
  size_t out_sizeof_element = 0;
  if (datum.encoded()) {
    Dtype *original_data_cpu_ptr = original_data.mutable_cpu_data();
    Copy(datum, original_data_cpu_ptr, out_sizeof_element);
    original_data_gpu_ptr = original_data.mutable_gpu_data();
  } else {
    original_data_gpu_ptr = original_data.mutable_gpu_data();
    Copy(datum, original_data_gpu_ptr, out_sizeof_element);
  }

  TransformGPU(1,
      datum.channels(),
      datum.height(),
      datum.width(),
      out_sizeof_element,
      original_data_gpu_ptr,
      transformed_data,
      randoms);
}

#endif


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

// do_mirror, h_off, w_off require that random values be passed in,
// because the random draws should have been taken in deterministic order
template<typename Dtype>
void DataTransformer<Dtype>::TransformPtrInt(Datum& datum,
    Dtype *transformed_data, const std::array<unsigned int, 3>& rand) {
  Transform(datum, transformed_data, rand);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformPtrEntry(shared_ptr<Datum> datum,
    Dtype *transformed_ptr,
    std::array<unsigned int, 3> rand,
    bool output_labels,
    Dtype *label) {
  // Get label from datum if needed
  if (output_labels) {
    *label = datum->label();
  }

  // If datum is encoded, decoded and transform the cv::image.
  if (datum->encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
    << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(*datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(*datum);
    }
    // Transform the cv::image into blob.
    TransformPtr(cv_img, transformed_ptr, rand);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    TransformPtrInt(*datum, transformed_ptr, rand);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
    TBlob<Dtype> *transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
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
    Transform(cv_img, transformed_blob);
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
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

  bool use_gpu_transform = param_.use_gpu_transform() && Caffe::mode() == Caffe::GPU;
  std::array<unsigned int, 3> rand;
  Fill3Randoms(&rand.front());
  if (use_gpu_transform) {
#ifndef CPU_ONLY
    Dtype *transformed_data_gpu = transformed_blob->mutable_gpu_data();
    TransformGPU(datum, transformed_data_gpu, rand);
    transformed_blob->cpu_data();
#else
  NO_GPU;
#endif
  } else {
    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    Transform(datum, transformed_data, rand);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum>& datum_vector,
    TBlob<Dtype> *transformed_blob) {
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

#ifdef USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat>& mat_vector,
    TBlob<Dtype> *transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
                         "The size of mat_vector must be equals to transformed_blob->num()";
  TBlob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    TBlob<Dtype> *transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype *transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar *ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
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
void DataTransformer<Dtype>::TransformPtr(const cv::Mat& cv_img,
    Dtype *transformed_ptr, const std::array<unsigned int, 3>& rand) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && (rand[0] % 2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  const float* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels)
        << "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  int height = img_height;
  int width = img_width;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    height = width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand[1] % (img_height - crop_size + 1);
      w_off = rand[2] % (img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  }

  CHECK(cv_cropped_img.data);

  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar *ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_ptr[top_index] =
              (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_ptr[top_index] =
                (pixel - mean_values_[c]) * scale;
          } else {
            transformed_ptr[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

#endif  // USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferDatumShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
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
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
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

#ifdef USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferCVMatShape(const cv::Mat& cv_img) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = img_height;
  shape[3] = img_width;
  return shape;
}

#endif  // USE_OPENCV

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
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum, bool use_gpu) {
  return InferBlobShape(InferDatumShape(datum), use_gpu);
}

#ifdef USE_OPENCV

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img, bool use_gpu) {
  return InferBlobShape(InferCVMatShape(cv_img), use_gpu);
}

#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() || (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    // Use random_seed setting for deterministic transformations
    const uint64_t random_seed = param_.random_seed() >= 0 ?
        static_cast<uint64_t>(param_.random_seed()) : Caffe::next_seed();
    rng_.reset(new Caffe::RNG(random_seed));
  } else {
    rng_.reset();
  }
}

template<typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() const {
  CHECK(rng_);
  caffe::rng_t *rng =
      static_cast<caffe::rng_t *>(rng_->generator());
  // this doesn't actually produce a uniform distribution
  return static_cast<unsigned int>((*rng)());
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
