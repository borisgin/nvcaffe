#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include <turbojpeg.h>

#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output)) << "Possible reasons: no disk space, "
      "no write permissions, the destination folder doesn't exist";
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

/**
 * Decode Datum to cv::Mat
 * @param datum
 * @param color_mode -1 enforce gray, 0 deduce from datum, +1 enforce color
 * @param out
 */
vector<int> DecodeDatumToCVMat(const Datum& datum, int color_mode, cv::Mat& cv_img,
    bool shape_only, bool accurate_jpeg) {
  CHECK(datum.encoded()) << "Datum not encoded";
  const std::string& content = datum.data();
  const size_t content_size = content.size();
  return Decode(reinterpret_cast<const unsigned char*>(content.data()), content_size,
                color_mode, &cv_img, nullptr, 0, shape_only, accurate_jpeg);
}

void DecodeDatumToSignedBuf(const Datum& datum, int color_mode,
    char* buf, size_t buf_len, bool accurate_jpeg) {
  CHECK(datum.encoded()) << "Datum not encoded";
  const std::string& content = datum.data();
  const size_t content_size = content.size();
  Decode(reinterpret_cast<const unsigned char*>(content.data()), content_size,
         color_mode, nullptr, buf, buf_len, false, accurate_jpeg);
}

// decodes to either cv_img or buf
vector<int> Decode(const unsigned char* content, size_t content_size, int color_mode,
    cv::Mat* cv_img, char* buf, size_t buf_len, bool shape_only, bool accurate_jpeg) {
  if (content_size > 1
      && content[0] == 255
      && content[1] == 216) {  // probably jpeg
    int width, height, subsamp;
    auto *content_data = const_cast<unsigned char *>(content);

    tjhandle jpeg_decoder = tjInitDecompress();
    tjDecompressHeader2(jpeg_decoder, content_data, content_size, &width, &height, &subsamp);

    int ch = subsamp == TJSAMP_GRAY ? 1 : 3;
    if (color_mode < 0) {
      ch = 1;
    }
    if (!shape_only) {
      if (cv_img != nullptr) {
        cv_img->create(height, width, ch == 3 ? CV_8UC3 : CV_8UC1);
      } else {
        CHECK_EQ(ch * height * width, buf_len);
      }
      if (0 > tjDecompress2(jpeg_decoder, content_data, content_size,
                            cv_img != nullptr ? cv_img->ptr<unsigned char>()
                                              : reinterpret_cast<unsigned char*>(buf),
                            width,
                            0,
                            height,
                            ch == 3 ? TJPF_BGR : TJPF_GRAY,  // TODO RGB?
                            (accurate_jpeg ? TJFLAG_ACCURATEDCT : TJFLAG_FASTDCT) |
                            TJFLAG_NOREALLOC)) {
        return vector<int>{};
      }
    }
    tjDestroy(jpeg_decoder);
    return vector<int>{1, ch, height, width};
  }
  // probably not jpeg...
  std::vector<char> vec_data(content, content + content_size);
  const int flag = color_mode < 0 ? cv::IMREAD_GRAYSCALE :
                   (color_mode > 0 ? cv::IMREAD_COLOR : cv::IMREAD_ANYCOLOR);
  if (cv_img != nullptr && !shape_only) {
    *cv_img = cv::imdecode(vec_data, flag);
    return vector<int>{1, cv_img->channels(), cv_img->rows, cv_img->cols};
  }
  cv::Mat img = cv::imdecode(vec_data, flag);
  if (!shape_only) {
    CHECK_EQ(img.channels() * img.rows * img.cols, buf_len);
    std::memcpy(buf, img.data, buf_len);  // NOLINT(caffe/alt_fn)
  }
  return vector<int>{1, img.channels(), img.rows, img.cols};
}

cv::Mat ReadImageToCVMat(const string& filename,
    int height, int width, bool is_color, int short_side) {
  cv::Mat cv_img_origin;
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  std::string content;
  ifs.seekg(0, std::ios::end);
  content.resize(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  ifs.read(&content.front(), content.size());
  ifs.close();

  cv::Mat cv_img;
  vector<int> shape = Decode(reinterpret_cast<const unsigned char*>(content.data()), content.size(),
                             is_color ? 1 : -1, &cv_img_origin, nullptr, 0, false, true);
  if (shape.size() == 0) {
    int cv_read_flag = is_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
    // Trying this one. It's slow but might decode better
    cv_img_origin = cv::imread(filename, cv_read_flag);
  }
  if (cv_img_origin.data) {
    if (is_color && cv_img_origin.channels() < 3) {
      cv::cvtColor(cv_img_origin, cv_img, CV_GRAY2RGB);
    }
    if (short_side > 0) {
      if (cv_img_origin.rows > cv_img_origin.cols) {
        width = short_side;
        height = cv_img_origin.rows * short_side / cv_img_origin.cols;
      } else {
        height = short_side;
        width = cv_img_origin.cols * short_side / cv_img_origin.rows;
      }
    }
    if (height <= 0 || width <= 0) {
      return cv_img.data ? cv_img : cv_img_origin;
    }
    cv::Size sz(width, height);
    if (cv_img.data) {
      cv::resize(cv_img, cv_img_origin, sz, 0., 0., CV_INTER_LINEAR);
      return cv_img_origin;
    }
    cv::resize(cv_img_origin, cv_img, sz, 0., 0., CV_INTER_LINEAR);
  } else {
    LOG(ERROR) << "Could not decode file " << filename;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) ) {
        return ReadFileToDatum(filename, label, datum);
      }
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, *datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

// tests only, TODO: clean
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  DecodeDatumToCVMat(datum, 0, cv_img, false);
  return cv_img;
}

// tests only, TODO: clean
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  DecodeDatumToCVMat(datum, is_color ? 1 : -1, cv_img, false);
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img;
    DecodeDatumToCVMat(*datum, 0, cv_img, false);
    CVMatToDatum(cv_img, *datum);
    return true;
  } else {
    return false;
  }
}

bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img;
    DecodeDatumToCVMat(*datum, is_color ? 1 : 0, cv_img, false);
    CVMatToDatum(cv_img, *datum);
    return true;
  } else {
    return false;
  }
}

vector<int> DatumToCVMat(const Datum& datum, cv::Mat& img, bool shape_only) {
  if (datum.encoded()) {
    LOG(FATAL) << "Datum encoded";
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  if (shape_only) {
    return vector<int>{1, datum_channels, datum_height, datum_width};
  }
  const int datum_size = datum_channels * datum_height * datum_width;
  CHECK_GT(datum_channels, 0);
  CHECK_GT(datum_height, 0);
  CHECK_GT(datum_width, 0);
  img.create(datum_height, datum_width, CVFC<float>(datum_channels));
  CHECK_EQ(img.channels(), datum_channels);
  CHECK_EQ(img.rows, datum_height);
  CHECK_EQ(img.cols, datum_width);
  const std::string& datum_buf = datum.data();
  CHECK_EQ(datum_buf.size(), datum_size);
  // CHW -> HWC
  chw2hwc(datum_channels, datum_width, datum_height,
      reinterpret_cast<const unsigned char*>(&datum_buf.front()), img.ptr<float>(0));
  return vector<int>{1, datum_channels, datum_height, datum_width};
}


void CVMatToDatum(const cv::Mat& cv_img, Datum& datum) {
  const unsigned int img_channels = cv_img.channels();
  const unsigned int img_height = cv_img.rows;
  const unsigned int img_width = cv_img.cols;
  const unsigned int img_size = img_channels * img_height * img_width;
  CHECK_GT(img_channels, 0);
  CHECK_GT(img_height, 0);
  CHECK_GT(img_width, 0);
  string* buf = datum.release_data();
  if (buf == nullptr || buf->size() != img_size) {
    delete buf;
    buf = new string(img_size, 0);
  }
  unsigned char* buf_front = reinterpret_cast<unsigned char*>(&buf->front());
  // HWC -> CHW
  if (cv_img.depth() == CV_8U) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<unsigned char>(0), buf_front);
  } else if (cv_img.depth() == CV_32F) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<float>(0), buf_front);
  } else if (cv_img.depth() == CV_64F) {
    hwc2chw(img_channels, img_width, img_height, cv_img.ptr<double>(0), buf_front);
  }
  datum.set_allocated_data(buf);
  datum.set_channels(img_channels);
  datum.set_height(img_height);
  datum.set_width(img_width);
  datum.set_encoded(false);
}

}  // namespace caffe
