#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

Timer::Timer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false),
      start_gpu_(nullptr),
      stop_gpu_(nullptr),
      device_(-1) {
  Init();
}

Timer::~Timer() {
  if (Caffe::mode() == Caffe::GPU) {
    int current_device;  // Just to check CUDA status:
    cudaError_t status = cudaGetDevice(&current_device);
    // Preventing crash while Caffe shutting down.
    if (status != cudaErrorCudartUnloading) {
      if (start_gpu_ != nullptr) {
        CUDA_CHECK(cudaEventDestroy(start_gpu_));
      }
      if (stop_gpu_ != nullptr) {
        CUDA_CHECK(cudaEventDestroy(stop_gpu_));
      }
    }
  }
}

void Timer::Start() {
  if (!running()) {
    if (Caffe::mode() == Caffe::GPU) {
      CHECK_EQ(device_, Caffe::current_device());
      CUDA_CHECK(cudaEventRecord(start_gpu_, 0));
    } else {
      start_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
    if (Caffe::mode() == Caffe::GPU) {
      CHECK_EQ(device_, Caffe::current_device());
      CUDA_CHECK(cudaEventRecord(stop_gpu_, 0));
      CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
    } else {
      stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    }
    running_ = false;
  }
}

float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
    CHECK_EQ(device_, Caffe::current_device());
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_, stop_gpu_));
    // Cuda only measure milliseconds
    elapsed_microseconds_ = elapsed_milliseconds_ * 1000;
  } else {
    elapsed_microseconds_ = (stop_cpu_ - start_cpu_).total_microseconds();
  }
  return elapsed_microseconds_;
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  if (Caffe::mode() == Caffe::GPU) {
    CHECK_EQ(device_, Caffe::current_device());
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_, stop_gpu_));
  } else {
    elapsed_milliseconds_ = (stop_cpu_ - start_cpu_).total_milliseconds();
  }
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
    if (Caffe::mode() == Caffe::GPU) {
      int current_device = Caffe::current_device();
      if (device_ < 0) {
        device_ = current_device;
      } else {
        CHECK_EQ(device_, current_device);
      }
      CUDA_CHECK(cudaEventCreate(&start_gpu_));
      CUDA_CHECK(cudaEventCreate(&stop_gpu_));
    } else {
      start_gpu_ = nullptr;
      stop_gpu_ = nullptr;
    }
    initted_ = true;
  }
}

CPUTimer::CPUTimer() {
  this->initted_ = true;
  this->running_ = false;
  this->has_run_at_least_once_ = false;
}

void CPUTimer::Start() {
  if (!running()) {
    this->start_cpu_ = boost::posix_time::microsec_clock::local_time();
    this->running_ = true;
    this->has_run_at_least_once_ = true;
  }
}

void CPUTimer::Stop() {
  if (running()) {
    this->stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    this->running_ = false;
  }
}

float CPUTimer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_milliseconds_ = (this->stop_cpu_ -
                                this->start_cpu_).total_milliseconds();
  return this->elapsed_milliseconds_;
}

float CPUTimer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    LOG(WARNING) << "Timer has never been run before reading time.";
    return 0;
  }
  if (running()) {
    Stop();
  }
  this->elapsed_microseconds_ = (this->stop_cpu_ -
                                this->start_cpu_).total_microseconds();
  return this->elapsed_microseconds_;
}

}  // namespace caffe
