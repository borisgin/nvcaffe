#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template<typename Dtype>
class DataTransformer {
 public:
  DataTransformer(const TransformationParameter& param, Phase phase);
  ~DataTransformer() = default;

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

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
  vector<int> Transform(const Datum* datum, Dtype* buf, size_t buf_len);

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
  void Transform(const cv::Mat& src, Dtype* buf, size_t buf_len);

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
  void Transform(const vector<cv::Mat>& mat_vector, TBlob<Dtype>* transformed_blob);
  void Transform(const cv::Mat& cv_img, TBlob<Dtype> *transformed_blob);

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
  void Transform(const vector<Datum>& datum_vector, TBlob<Dtype>* transformed_blob);

  void Transform(Datum& datum, TBlob<Dtype>* transformed_blob);
  void VariableSizedTransforms(Datum* datum);

 protected:
  bool image_random_resize_enabled() const;
  bool image_random_crop_enabled() const;
  bool image_center_crop_enabled() const;

  void apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst);
  void image_random_crop(int crop_w, int crop_h, cv::Mat& img);
  void TransformV1(const Datum& datum, Dtype* buf, size_t buf_len);

  static void image_random_resize(int new_size, const cv::Mat& src, cv::Mat& dst);
  static void image_center_crop(int crop_w, int crop_h, cv::Mat& img);

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

 protected:
  unsigned int Rand() const;

  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  TBlob<float> data_mean_;
  vector<float> mean_values_;

  cv::Mat mean_mat_orig_, mean_mat_;
  cv::Mat tmp_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
