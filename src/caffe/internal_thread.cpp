#include <boost/thread.hpp>
#include <exception>
#include <caffe/caffe.hpp>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::InternalThread(int target_device, size_t rank, size_t threads, bool delayed)
    : target_device_(target_device),
      rank_(rank),
      aux_(nullptr),
      threads_(threads),
      delay_flags_(threads, make_shared<Flag>(!delayed)) {}

void InternalThread::StartInternalThread(bool set_cpu_affinity) {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";
  LOG(INFO) << "Starting internal thread on device " << target_device_;
  Caffe::Brew mode = Caffe::mode();
  if (mode == Caffe::GPU) {
    CHECK_GE(target_device_, 0);
  }
  const int rand_seed = caffe_rng_rand();
  const int solver_count = Caffe::solver_count();
  try {
    for (size_t id = 0; id < threads_.size(); ++id) {
      threads_[id] = boost::thread(&InternalThread::entry, this, id, target_device_, mode,
          rand_seed, solver_count, rank_, set_cpu_affinity);
    }
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThread::RestartAllThreads(size_t new_threads, bool delayed, bool set_cpu_affinity) {
  if (new_threads == 0UL) {
    return;
  }
  LOG(INFO) << "Restarting " << new_threads << " internal thread(s) on device " << target_device_;
  Caffe::Brew mode = Caffe::mode();
  if (mode == Caffe::GPU) {
    CHECK_GE(target_device_, 0);
  }
  const int rand_seed = caffe_rng_rand();
  const int solver_count = Caffe::solver_count();
  CHECK_EQ(1, threads_.size());
  threads_.clear();
  delay_flags_.clear();
  threads_.resize(new_threads);
  delay_flags_.resize(new_threads);
  try {
    for (size_t id = 0; id < new_threads; ++id) {
      delay_flags_[id] = make_shared<Flag>(!delayed);
      threads_[id] = boost::thread(&InternalThread::entry, this, id,
          target_device_, mode, rand_seed, solver_count, rank_, set_cpu_affinity);
    }
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThread::entry(int thread_id, int device, Caffe::Brew mode, int rand_seed,
    int solver_count, size_t rank, bool set_cpu_affinity) {
  delay_flags_[thread_id]->wait();
  if (mode == Caffe::GPU) {
    CHECK_GE(device, 0);
  }
  rank_ = rank;  // TODO ?
  target_device_ = device;
  LOG(INFO) << "Started internal thread " << std::this_thread::get_id()
            << " on device " << device << ", rank " << rank_;
#ifndef CPU_ONLY
  if (mode == Caffe::GPU) {
    CUDA_CHECK(cudaSetDevice(device));
    if (set_cpu_affinity) {
#ifndef NO_NVML
      nvml::setCpuAffinity(rank);
#endif
    }
  }
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);

  if (threads_.size() == 1) {
    InternalThreadEntry();
  } else {
    InternalThreadEntryN(thread_id);
  }
}

void InternalThread::StopInternalThread() {
  for (size_t id = 0; id < threads_.size(); ++id) {
    if (is_started(id)) {
      threads_[id].interrupt();
    }
  }
  WaitAll();
}

void InternalThread::WaitAll() {
  try {
    for (size_t id = 0; id < threads_.size(); ++id) {
      if (is_started(id)) {
        threads_[id].join();
      }
    }
  } catch (boost::thread_interrupted&) {
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

}  // namespace caffe
