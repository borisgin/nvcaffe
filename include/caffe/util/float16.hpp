#ifndef CAFFE_UTIL_FP16_H_
#define CAFFE_UTIL_FP16_H_

#include <algorithm>
#include <cfloat>
#include <iosfwd>
#include <glog/logging.h>

#ifdef CPU_ONLY
  #define CAFFE_UTIL_HD
  #define CAFFE_UTIL_IHD inline
#else
  #include "caffe/util/fp16_emu.h"
  #define CAFFE_UTIL_HD __host__ __device__
  #define CAFFE_UTIL_IHD __inline__ __host__ __device__
  #include "half_float/half.hpp"

namespace caffe {
  typedef half_float::half float16;
}   // namespace caffe

#endif
#endif
