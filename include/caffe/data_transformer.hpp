#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
class DataTransformer {
 public:
  DataTransformer(const TransformationParameter& param, Phase phase);
  ~DataTransformer() = default;

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  template<typename Dtype>
  void TransformGPU(int N, int C, int H, int W, size_t sizeof_element,
      const void* in, Dtype* out, const unsigned int* rands, bool signed_data);

  /**
   * @brief Applies transformations defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum [in]
   *    The source Datum containing data of arbitrary shape.
   * @param buf_len [in]
   *    Buffer length in Dtype elements
   * @param buf [out]
   *    The destination array that will store transformed data of a fixed
   *    shape. If nullptr passed then only shape vector is computed.
   * @return Output shape
   */
  template<typename Dtype>
  vector<int> Transform(const Datum* datum, Dtype* buf, size_t buf_len,
      Packing& out_packing, bool repack = true) {
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

  /**
   * @brief Applies transformations defined in the image data layer's
   * transform_param block to the data.
   *
   * @param datum [in]
   *    The source cv::Mat containing data of arbitrary shape.
   * @param buf_len [in]
   *    Buffer length in Dtype elements
   * @param buf [out]
   *    The destination array that will store transformed data of a fixed
   *    shape.
   */
  template<typename Dtype>
  void Transform(const cv::Mat& src, Dtype* buf, size_t buf_len, bool repack = true) {
    cv::Mat tmp, dst;

    image_random_resize(src, tmp);

    if (image_random_crop_enabled()) {
      image_random_crop(param_.crop_size(), param_.crop_size(), tmp);  // TODO
    } else if (image_center_crop_enabled()) {
      image_center_crop(param_.crop_size(), param_.crop_size(), tmp);
    }
    apply_mean_scale_mirror(tmp, dst);
    FloatCVMatToBuf<Dtype>(dst, buf_len, buf, repack);
  }

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  template<typename Dtype>
  void Transform(const vector<cv::Mat>& mat_vector, TBlob<Dtype>* transformed_blob) {
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
  void Transform(const cv::Mat& img, TBlob<Dtype> *transformed_blob) {
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

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  template<typename Dtype>
  void Transform(const vector<Datum>& datum_vector, TBlob<Dtype>* transformed_blob) {
    const size_t datum_num = datum_vector.size();
    const int num = transformed_blob->num();
    CHECK_GT(datum_num, 0) << "There is no datum to add";
    CHECK_LE(datum_num, num)
      << "The size of datum_vector must be not greater than transformed_blob->num()";
    cv::Mat dst;
    size_t buf_len = transformed_blob->offset(1);
    for (size_t item_id = 0; item_id < datum_num; ++item_id) {
      size_t offset = transformed_blob->offset(item_id);
      DatumToCVMat(datum_vector[item_id], dst, false);
      FloatCVMatToBuf<Dtype>(dst, buf_len, transformed_blob->mutable_cpu_data(false) + offset);
    }
  }

  /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  unsigned int Rand(int n) const {
    CHECK_GT(n, 0);
    return Rand() % n;
  }

  // tests only, TODO: clean
  template<typename Dtype>
  void Transform(Datum& datum, TBlob<Dtype>* transformed_blob) {
    cv::Mat img;
    DatumToCVMat(datum, img, false);
    Transform(img, transformed_blob);
  }

  void VariableSizedTransforms(Datum* datum);
  void Fill3Randoms(unsigned int *rand) const;

 protected:
  bool image_random_resize_enabled() const;
  bool image_random_crop_enabled() const;
  bool image_center_crop_enabled() const;

  void apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst);
  void image_random_crop(int crop_w, int crop_h, cv::Mat& img);

  template<typename Dtype>
  void TransformV1(const Datum& datum, Dtype* buf, size_t buf_len) {
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

  void image_random_resize(const cv::Mat& src, cv::Mat& dst);
  static void image_center_crop(int crop_w, int crop_h, cv::Mat& img);
  unsigned int Rand() const;
  float Rand(float lo, float up) const;

  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  TBlob<float> data_mean_;
  vector<float> mean_values_;
  cv::Mat mean_mat_orig_, mean_mat_;
  cv::Mat tmp_;

  const float rand_resize_ratio_lower_, rand_resize_ratio_upper_;
  const float vertical_stretch_lower_;
  const float vertical_stretch_upper_;
  const float horizontal_stretch_lower_;
  const float horizontal_stretch_upper_;
  const bool allow_upscale_;
  GPUMemory::Workspace mean_values_gpu_;

  static constexpr double UM = static_cast<double>(UINT_MAX);
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
