#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

template<typename Dtype>
inline int CVFC(int ch);

template<>
inline int CVFC<float>(int ch) {
  return CV_32FC(ch);
}

template<>
inline int CVFC<double>(int ch) {
  return CV_64FC(ch);
}

using ::google::protobuf::Message;
using ::boost::filesystem::path;

// HWC -> CHW
template <typename Stype, typename Dtype>
inline void hwc2chw(size_t ch, size_t w, size_t h, const Stype* src, Dtype* dst) {
  size_t index = 0UL;
  const size_t hw_stride = w * h;
  for (size_t s = 0UL; s < hw_stride; ++s) {
    size_t stride_index = s;
    for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
      dst[stride_index] = static_cast<Dtype>(src[index++]);
    }
  }
}

// CHW -> HWC
template <typename Stype, typename Dtype>
inline void chw2hwc(size_t ch, size_t w, size_t h, const Stype* src, Dtype* dst) {
  size_t index = 0UL;
  const size_t hw_stride = w * h;
  for (size_t s = 0UL; s < hw_stride; ++s) {
    size_t stride_index = s;
    for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
      dst[index++] = static_cast<Dtype>(src[stride_index]);
    }
  }
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);
bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);
void CVMatToDatum(const cv::Mat& cv_img, Datum& datum);
vector<int> DatumToCVMat(const Datum& datum, cv::Mat& img, bool shape_only);
vector<int> DecodeDatumToCVMat(const Datum& datum, int color_mode, cv::Mat& cv_img,
    bool shape_only, bool accurate_jpeg = true);

template<typename Dtype>
void TBlobDataToCVMat(const TBlob<Dtype>& blob, cv::Mat& img) {
  const int blob_channels = blob.channels();
  const int blob_height = blob.height();
  const int blob_width = blob.width();
  const int blob_size = blob_channels * blob_height * blob_width;
  CHECK_EQ(blob.count(), blob_size);
  CHECK_GT(blob_channels, 0);
  CHECK_GT(blob_height, 0);
  CHECK_GT(blob_width, 0);
  img.create(blob_height, blob_width, CVFC<float>(blob_channels));
  CHECK_EQ(img.channels(), blob_channels);
  CHECK_EQ(img.rows, blob_height);
  CHECK_EQ(img.cols, blob_width);
  const Dtype* blob_buf = blob.cpu_data();
  // CHW -> HWC
  chw2hwc(blob_channels, blob_width, blob_height, blob_buf, img.ptr<float>(0));
}

template<typename Dtype>
void FloatCVMatToBuf(const cv::Mat& cv_img, size_t buf_len, Dtype* buf) {
  const size_t img_channels = cv_img.channels();
  const size_t img_height = cv_img.rows;
  const size_t img_width = cv_img.cols;
  CHECK_GT(img_channels, 0UL);
  CHECK_GT(img_height, 0UL);
  CHECK_GT(img_width, 0UL);
  const size_t img_size = img_channels * img_height * img_width;
  CHECK_LE(img_size, buf_len);
  // TODO This is the place where we fill top blob
  // Here we might leave HWC as is for faster convolutions
  // So far, we do slow HWC -> CHW transformation
  if (cv_img.depth() == CV_8U) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<unsigned char>(0), buf);
  } else if (cv_img.depth() == CV_32F) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<float>(0), buf);
  } else if (cv_img.depth() == CV_64F) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<double>(0), buf);
  }
//  cv::Mat im;
//  im.create(img_height, img_width, CVFC<float>(img_channels));
//  chw2hwc(img_channels, img_width, img_height, buf, im.ptr<float>(0));
//  cv::Mat dsp;
////  im.convertTo(dsp, CV_32F);
//  cv::normalize(im, dsp, 0, 1, cv::NORM_MINMAX);
//  cv::imshow("test", dsp);
//  cv::waitKey(0);
}

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
