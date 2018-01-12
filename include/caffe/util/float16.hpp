#ifndef CAFFE_UTIL_FP16_H_
#define CAFFE_UTIL_FP16_H_

#include <algorithm>
#include <cfloat>
#include <iosfwd>
#include <glog/logging.h>

#ifdef __CUDACC__
#include "caffe/util/half.cuh"
#endif
#include "half_float/half.hpp"

namespace caffe {
  typedef half_float::half float16;
}

#endif
