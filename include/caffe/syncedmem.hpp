#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include <boost/thread.hpp>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  explicit SyncedMemory(size_t size = 0UL)
      : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        device_(-1), valid_(true) {}
  ~SyncedMemory();
  const void* cpu_data();
  const void* gpu_data(int group = 0);
  void set_cpu_data(void* data);
  void set_gpu_data(void* data);
  void* mutable_cpu_data(bool copy_from_gpu = true);
  void* mutable_gpu_data(bool copy_from_cpu = true, int group = 0);

  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };

  SyncedHead head() const {
    return head_;
  }
  size_t size() const {
    return size_;
  }
  void resize(size_t size) {
    size_ = size;
  }
  size_t gpu_memory_use(bool own_only = false) const {
    return own_only ? (own_gpu_data_ ? size_ : 0UL) : size_;
  }
  bool is_valid() const {
    return valid_;
  }
  void invalidate() {
    valid_ = false;
  }
  void validate() {
    valid_ = true;
  }

  std::string to_string(int indent, Type type);  // debug helper

 protected:
  void MallocHost(void** ptr, size_t size, bool* use_cuda);
  void FreeHost(void* ptr, bool use_cuda);

 private:
  void to_cpu(bool copy_from_gpu = true);
  void to_gpu(bool copy_from_cpu = true, int group = 0);

  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int  device_;
  bool valid_;
  shared_ptr<CudaStream> pstream_;

  DISABLE_COPY_MOVE_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
