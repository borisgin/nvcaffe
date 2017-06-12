#ifndef CAFFE_UTIL_FP16_H_
#define CAFFE_UTIL_FP16_H_

#include <algorithm>
#include <cfloat>
#include <iosfwd>
#include <glog/logging.h>

#ifndef CPU_ONLY

#define HLF_EPSILON  4.887581E-04
#define HLF_MIN      6.103516E-05
#define HLF_MAX      6.550400E+04
#define HLF_TRUE_MIN 5.960464E-08

//  #include "caffe/util/half.cuh"
  #include "half_float/half.hpp"

namespace caffe {
  typedef half_float::half float16;
}

#endif
#endif
