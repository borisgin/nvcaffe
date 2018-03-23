#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/float16.hpp"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#if !defined(CUDNN_VERSION) || !CUDNN_VERSION_MIN(6, 0, 0)
#error "NVCaffe 0.16 and higher requires CuDNN version 6.0.0 or higher"
#endif

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status) << ", device " << Caffe::current_device(); \
  } while (0)

#define CUDNN_CHECK2(condition, arg1, arg2) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << "CuDNN error " \
      << (int)status << " " << (arg1) << " " << (arg2); \
  } while (0)

namespace caffe {

namespace cudnn {

template<typename Dtype>
class dataType;

template<>
class dataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static const cudnnDataType_t conv_type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template<>
class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static const cudnnDataType_t conv_type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template<>
class dataType<float16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
  static const cudnnDataType_t conv_type = CUDNN_DATA_HALF;
  static float oneval, zeroval;
  static const void *one, *zero;
};

inline
const void* one(Type type) {
  const void* ret = nullptr;
  switch (type) {
    case FLOAT:
      ret = dataType<float>::one;
      break;
    case FLOAT16:
      ret = dataType<float16>::one;
      break;
    case DOUBLE:
      ret = dataType<double>::one;
      break;
    default:
      LOG(FATAL) << "Unknown Type " << Type_Name(type);
      break;
  }
  return ret;
}

inline
const void* zero(Type type) {
  const void* ret = nullptr;
  switch (type) {
    case FLOAT:
      ret = dataType<float>::zero;
      break;
    case FLOAT16:
      ret = dataType<float16>::zero;
      break;
    case DOUBLE:
      ret = dataType<double>::zero;
      break;
    default:
      LOG(FATAL) << "Unknown Type " << Type_Name(type);
      break;
  }
  return ret;
}

inline
cudnnDataType_t cudnn_data_type(Type math) {
  cudnnDataType_t ret;
  switch (math) {
    case FLOAT:
      ret = dataType<float>::conv_type;
      break;
    case FLOAT16:
      if (caffe::Caffe::device_capability(caffe::Caffe::current_device()) >= 600) {
        ret = dataType<float16>::conv_type;
      } else {
        ret = dataType<float>::conv_type;
      }
      break;
    case DOUBLE:
      ret = dataType<double>::conv_type;
      break;
    default:
      LOG(FATAL) << "Unknown Math type " << Type_Name(math);
      break;
  }
  return ret;
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc, int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, cudnn::dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
}

inline void setConvolutionDesc(Type math, cudnnConvolutionDescriptor_t conv,
      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
  int padA[2] = {pad_h, pad_w};
  int strideA[2] = {stride_h, stride_w};
  int upscaleA[2] = {dilation_h, dilation_w};
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv, 2, padA, strideA, upscaleA,
      CUDNN_CROSS_CORRELATION, cudnn::cudnn_data_type(math)));
}

template<typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t *desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template<typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
      n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template<typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
      stride_n, stride_c, stride_h, stride_w);
}

inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc, cudnnDataType_t type,
    Packing packing, const vector<int> &shape) {
  int stride_w = 0, stride_h = 0, stride_c = 0, stride_n = 0;
  const int n = shape[0];
  const int c = shape[1];
  const int h = shape[2];
  const int w = shape[3];
  if (packing == NCHW) {
    stride_w = 1;
    stride_h = w * stride_w;
    stride_c = h * stride_h;
    stride_n = c * stride_c;
  } else if (packing == NHWC) {
    stride_c = 1;
    stride_w = c * stride_c;
    stride_h = w * stride_w;
    stride_n = h * stride_h;
  } else {
    LOG(FATAL) << "Unknown packing";
  }
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, type,
      n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

inline void setTensor4dDesc(cudnnTensorDescriptor_t *desc, Type type,
    Packing packing, const vector<int> &shape) {
  setTensor4dDesc(desc, cudnn_data_type(type), packing, shape);
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
