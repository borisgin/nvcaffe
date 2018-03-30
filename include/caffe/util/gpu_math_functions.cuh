#ifndef INCLUDE_CAFFE_GPU_MATH_FUNCTIONS_H_
#define INCLUDE_CAFFE_GPU_MATH_FUNCTIONS_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <glog/logging.h>
#include "math_functions.h"
#include "driver_types.h"
#include "caffe/common.hpp"
#include "caffe/util/half.cuh"

__device__ __inline__
bool hlt(half a, half b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hlt(a, b);
#else
  return __half2float(a) < __half2float(b);
#endif
}

__device__ __inline__
half hmul(half a, half b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hmul(a, b);
#else
  return float2half_clip(__half2float(a) * __half2float(b));
#endif
}

__device__ __inline__
half hdiv(half a, half b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hdiv(a, b);
#else
  return float2half_clip(__half2float(a) / __half2float(b));
#endif
}

__device__ __inline__
half2 hmul2(half2 a, half2 b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hmul2(a, b);
#else
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);

  af.x *= bf.x;
  af.y *= bf.y;

  return float22half2_clip(af);
#endif
}

__device__ __inline__
half2 hge2(half2 a, half2 b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hge2(a, b);
#else
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);

  af.x = (af.x >= bf.x) ? 1.f : 0.f;
  af.y = (af.y >= bf.y) ? 1.f : 0.f;

  return float22half2_clip(af);
#endif
}

__device__ __inline__
half hadd(half a, half b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hadd(a, b);
#else
  return float2half_clip(__half2float(a) + __half2float(b));
#endif
}

__device__ __inline__
half2 hadd2(half2 a, half2 b) {
#if __CUDA_ARCH__ >= 530 && !defined(OLD_CUDA_HALF_IMPL)
  return __hadd2(a, b);
#else
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);

  af.x += bf.x;
  af.y += bf.y;

  return float22half2_clip(af);
#endif
}

__device__ __inline__
half h_abs(half a) {
  a.setx(a.x() & 0x7FFF);
  return a;
}

__device__ __inline__
half2 h2_abs(half2 a) {
#ifdef OLD_CUDA_HALF_IMPL
  a.x &= 0x7FFF7FFFU;
#else
  a.set_lo(h_abs(a.lo()));
  a.set_hi(h_abs(a.hi()));
#endif
  return a;
}

// a <- max(a,b)
__device__ __inline__
void h2_max_replace(volatile half2 *a, half2 b) {
  half2 cmp = hge2(*const_cast<half2*>(a), b);  // 00 01 10 11
  bool al = cmp.lo();                // true: a.low >= b.low
  bool ah = cmp.hi();               // true: a.high >= b.high
  if (al) {
    if (ah) {
      // (a.low,a.high)
    } else {
      // (a.low,b.high)
      const_cast<half2*>(a)->set_hi(b.hi());
    }
  } else {
    if (ah) {
      // (b.low,a.high)
      const_cast<half2*>(a)->set_lo(b.lo());
    } else {
      // (b.low,b.high)
      *const_cast<half2*>(a) = b;
    }
  }
}

// <- max(a,b)
__device__ __inline__
half2 h2_max(half2 a, half2 b) {
  half2 m = a;
  h2_max_replace(&m, b);
  return m;
}


template<typename T>
__device__ __inline__
T tabs(T t);
template<>
__device__ __inline__
int tabs<int>(int t) {
  return abs(t);
}
template<>
__device__ __inline__
float tabs<float>(float t) {
  return fabs(t);
}
template<>
__device__ __inline__
double tabs<double>(double t) {
  return fabs(t);
}
template<>
__device__ __inline__
half2 tabs<half2>(half2 t) {
  return h2_abs(t);
}

template<typename T>
__device__ __inline__
T tzero() {
  return 0;
}
template<>
__device__ __inline__
half2 tzero<half2>() {
  return half2(half(), half());
}

template<typename T>
__device__ __inline__
int non_zero_count(T a) {
  return a == 0 ? 0 : 1;
}
template<>
__device__ __inline__
int non_zero_count<float>(float a) {
  return a == 0.F ? 0 : 1;
}
template<>
__device__ __inline__
int non_zero_count<double>(double a) {
  return a == 0. ? 0 : 1;  // DBL_EPSILON?
}
template<>
__device__ __inline__
int non_zero_count<half2>(half2 a) {
  return (a.lo() ? 1 : 0) + (a.hi() ? 1 : 0);
}

template<typename T, typename TR>
__device__ __inline__
TR tmax(T a, T b) {
  return TR(a > b ? a : b);
}
template<>
__device__ __inline__
half2 tmax<half2, half2>(half2 a, half2 b) {
  return h2_max(a, b);
}
template<>
__device__ __inline__
float tmax<half2, float>(half2 a, half2 b) {
  float2 fm = __half22float2(h2_max(a, b));
  return tmax<float, float>(fm.x, fm.y);
}
template<>
__device__ __inline__
double tmax<half2, double>(half2 a, half2 b) {
  float2 fm = __half22float2(h2_max(a, b));
  return tmax<float, float>(fm.x, fm.y);
}

template<typename T, typename TR>
__device__ __inline__
TR tsumsq(T a, T b) {
  return TR(a * a) + TR(b * b);
}
template<>
__device__ __inline__
half2 tsumsq<half2, half2>(half2 a, half2 b) {
  return hadd2(hmul2(a, a), hmul2(b, b));
}
template<>
__device__ __inline__
float tsumsq<half2, float>(half2 a, half2 b) {
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);
  return af.x * af.x + af.y * af.y + bf.x * bf.x + bf.y * bf.y;
}
template<>
__device__ __inline__
double tsumsq<half2, double>(half2 a, half2 b) {
  return tsumsq<half2, float>(a, b);
}

template<typename T, typename TR>
__device__ __inline__
TR tsum(T a, T b) {
  return TR(a + b);
}
template<>
__device__ __inline__
half2 tsum<half2, half2>(half2 a, half2 b) {
  return hadd2(a, b);
}
template<>
__device__ __inline__
float tsum<half2, float>(half2 a, half2 b) {
  float2 af = __half22float2(a);
  float2 bf = __half22float2(b);
  return af.x + af.y + bf.x + bf.y;
}
template<>
__device__ __inline__
double tsum<half2, double>(half2 a, half2 b) {
  return tsum<half2, float>(a, b);
}

template<typename T>
__device__ __inline__
void tassign(volatile T *a, T b) {
  *a = b;
}

template<typename T, typename TR>
__device__ __inline__
void tmax_replace(volatile TR *a, T b) {
  if (b > *a) {
    *a = b;
  }
}
template<>
__device__ __inline__
void tmax_replace<half2, half2>(volatile half2 *a, half2 b) {
  h2_max_replace(a, b);
}
template<>
__device__ __inline__
void tmax_replace<half2, float>(volatile float *a, half2 b) {
  float2 f = __half22float2(b);
  if (f.x > f.y) {
    tmax_replace(a, f.x);
  } else {
    tmax_replace(a, f.y);
  }
}
template<>
__device__ __inline__
void tmax_replace<half2, double>(volatile double *a, half2 b) {
  float2 f = __half22float2(b);
  if (f.x > f.y) {
    tmax_replace(a, f.x);
  } else {
    tmax_replace(a, f.y);
  }
}

template<typename T, typename TR>
__device__ __inline__
void tsum_replace(volatile TR *a, T b) {
  *a += b;
}
template<>
__device__ __inline__
void tsum_replace<half2, half2>(volatile half2 *a, half2 b) {
  *const_cast<half2*>(a) = hadd2(*const_cast<half2*>(a), b);
}
template<>
__device__ __inline__
void tsum_replace<half2, float>(volatile float *a, half2 b) {
  float2 f = __half22float2(b);
  *a += f.x + f.y;
}
template<>
__device__ __inline__
void tsum_replace<half2, double>(volatile double *a, half2 b) {
  float2 f = __half22float2(b);
  *a += f.x + f.y;
}

template<typename T, typename TR>
__device__ __inline__
void tsumsq_replace(volatile TR *a, T b) {
  *a += TR(b * b);
}
template<>
__device__ __inline__
void tsumsq_replace<half2, half2>(volatile half2 *a, half2 b) {
  *const_cast<half2*>(a) = hadd2(*const_cast<half2*>(a), hmul2(b, b));
}
template<>
__device__ __inline__
void tsumsq_replace<half2, float>(volatile float *a, half2 b) {
  float2 f = __half22float2(b);
  *a += f.x * f.x + f.y * f.y;
}
template<>
__device__ __inline__
void tsumsq_replace<half2, double>(volatile double *a, half2 b) {
  float2 f = __half22float2(b);
  *a += f.x * f.x + f.y * f.y;
}


#define SHMEM(FN)  \
template<typename T>  \
struct __dyn_shmem_##FN##__ {  \
  __device__  \
  T *getPtr() {  \
    return NULL;  \
  }  \
};  \
  \
template<>  \
struct __dyn_shmem_##FN##__<float> {  \
  __device__  \
  float *getPtr() {  \
    extern __shared__ float S##FN##ptr[];  \
    return S##FN##ptr;  \
  }  \
};  \
  \
template<>  \
struct __dyn_shmem_##FN##__<double> {  \
  __device__  \
  double *getPtr() {  \
    extern __shared__ double D##FN##ptr[];  \
    return D##FN##ptr;  \
  }  \
};  \
  \
template<>  \
struct __dyn_shmem_##FN##__<half2> {  \
  __device__  \
  half2 *getPtr() {  \
    extern __shared__ half2 H##FN##ptr[];  \
    return H##FN##ptr;  \
  }  \
};  \
  \
template<>  \
struct __dyn_shmem_##FN##__<int> {  \
  __device__  \
  int *getPtr() {  \
    extern __shared__ int I##FN##ptr[];  \
    return I##FN##ptr;  \
  }  \
};  \
  \
template<>  \
struct __dyn_shmem_##FN##__<unsigned int> {  \
  __device__  \
  unsigned int *getPtr() {  \
    extern __shared__ unsigned int UI##FN##ptr[];  \
    return UI##FN##ptr;  \
  }  \
};  \
  \
template<int N>  \
struct n_bytes {  \
  char content[N];  \
}

#define SHMEMN(N, FN)                    \
template<> struct __dyn_shmem_##FN##__<n_bytes<N>> { \
  __device__ n_bytes<N>* getPtr() {           \
    extern __shared__ n_bytes<N> FN##ptr##N[];    \
    return FN##ptr##N;                            \
  }                                           \
}

#define CAFFE_GPU_SHMEM(FN)  \
SHMEMN(2, FN);  \
SHMEMN(4, FN);  \
SHMEMN(6, FN);  \
SHMEMN(8, FN);  \
SHMEMN(10, FN);  \
SHMEMN(12, FN);  \
SHMEMN(14, FN);  \
SHMEMN(16, FN)

#define REDUCTION_GROUPS_MAX 16

#endif  // INCLUDE_CAFFE_GPU_MATH_FUNCTIONS_H_
