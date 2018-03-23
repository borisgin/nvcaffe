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

#include "google/protobuf/repeated_field.h"
using google::protobuf::RepeatedPtrField;

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, subtracting the image mean...
 */
template<typename Dtype>
class DataTransformer {
 public:
  DataTransformer(const TransformationParameter& param, Phase phase);
  ~DataTransformer() = default;

  const TransformationParameter& transform_param() const {
    return param_;
  }

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

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
  vector<int> Transform(const Datum* datum, Dtype* buf, size_t buf_len,
      Packing& out_packing, bool repack = true);

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
  void Transform(const cv::Mat& src, Dtype* buf, size_t buf_len, bool repack = true) const;

  void Transform(const cv::Mat& img, TBlob<Dtype> *transformed_blob) const;

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
  void Transform(const vector<cv::Mat>& mat_vector, TBlob<Dtype>* transformed_blob) const;

  void Transform(const vector<Datum>& datum_vector, TBlob<Dtype> *transformed_blob) const;

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
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob) const {
    cv::Mat img;
    DatumToCVMat(datum, img, false);
    Transform(img, transformed_blob);
  }

  void Fill3Randoms(unsigned int *rand) const;

  void TransformInv(const Blob* blob, vector<cv::Mat>* cv_imgs);
  void TransformInv(const Dtype* data, cv::Mat* cv_img, const int height,
      const int width, const int channels);

  vector<int> InferBlobShape(const cv::Mat& cv_img);
  vector<int> InferDatumShape(const Datum& datum);
  vector<int> InferCVMatShape(const cv::Mat& img);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param bottom_shape
   *    The shape of the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<int>& bottom_shape, bool use_gpu = false);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);

  /**
   * @brief Crops the datum according to bbox.
   */

  void CropImage(const Datum& datum, const NormalizedBBox& bbox, Datum* crop_datum);

  /**
   * @brief Crops the datum and AnnotationGroup according to bbox.
   */
  void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                 AnnotatedDatum* cropped_anno_datum);

  /**
   * @brief Expand the datum.
   */
  void ExpandImage(const Datum& datum, const float expand_ratio,
                   NormalizedBBox* expand_bbox, Datum* expanded_datum);

  /**
   * @brief Expand the datum and adjust AnnotationGroup.
   */
  void ExpandImage(const AnnotatedDatum& anno_datum, AnnotatedDatum* expanded_anno_datum);

  /**
   * @brief Apply distortion to the datum.
   */
  void DistortImage(const Datum& datum, Datum* distort_datum);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the annotated data.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
   * @param transformed_anno_vec
   *    This is destination annotation.
   */
  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);

  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);

  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);

  void Transform(const AnnotatedDatum& anno_datum,
                 TBlob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec);

  bool image_random_resize_enabled() const;
  bool image_center_crop_enabled() const;
  bool image_random_crop_enabled() const;
  void image_random_resize(const cv::Mat& src, cv::Mat& dst) const;
  void image_center_crop(int crop_w, int crop_h, cv::Mat& img) const;
  void image_random_crop(int crop_w, int crop_h, cv::Mat& img) const;

 protected:
  void apply_mean_scale_mirror(const cv::Mat& src, cv::Mat& dst) const;

  void TransformV1(const Datum& datum, Dtype* buf, size_t buf_len);

  unsigned int Rand() const;
  float Rand(float lo, float up) const;

  void Copy(const Datum& datum, Dtype* data, size_t& out_sizeof_element);
  void Copy(const cv::Mat& datum, Dtype* data);

  /**
   * @brief Transform the annotation according to the transformation applied
   * to the datum.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param do_resize
   *    If true, resize the annotation accordingly before crop.
   * @param crop_bbox
   *    The cropped region applied to anno_datum.datum()
   * @param do_mirror
   *    If true, meaning the datum has mirrored.
   * @param transformed_anno_group_all
   *    Stores all transformed AnnotationGroup.
   */
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);


  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, TBlob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  /**
   * @brief Crops img according to bbox.
   */
  void CropImage(const cv::Mat& img, const NormalizedBBox& bbox, cv::Mat* crop_img);

  /**
   * @brief Expand img to include mean value as background.
   */
  void ExpandImage(const cv::Mat& img, const float expand_ratio,
                   NormalizedBBox* expand_bbox, cv::Mat* expand_img);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);

  void Transform(const Datum& datum,
      Dtype *transformed_data, const std::array<unsigned int, 3>& rand);

 protected:
  // Transform and return the transformation information.
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data and return transform information.
   */
  void Transform(const Datum& datum, TBlob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  TBlob<float> data_mean_;
  vector<float> mean_values_;
  cv::Mat mean_mat_orig_;
  mutable cv::Mat mean_mat_;
  mutable cv::Mat tmp_;

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
