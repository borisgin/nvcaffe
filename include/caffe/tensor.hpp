#ifndef INCLUDE_CAFFE_TENSOR_HPP_
#define INCLUDE_CAFFE_TENSOR_HPP_

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/type.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class Tensor {
  friend class Blob;

 public:
  explicit Tensor(Type type);

  ~Tensor() {}

 private:
  Type type() const {
    return type_;
  }

  void lock_tensor() {
    locked_ = true;
  }

  size_t size() const {
    return synced_arrays_->size();
  }

  void scale(float new_scale, void* handle = nullptr, bool synced = true);
  void cpu_scale(float new_scale);
  float cpu_amax();
  float cpu_asum();

#ifndef CPU_ONLY
  void gpu_set(float value, bool sync, cudaStream_t stream);
  void gpu_scale(float new_scale, cublasHandle_t cublas_handle, bool sync);
  float gpu_amax();
  size_t gpu_memory_use() const;
#endif

  void set(float value);
  size_t cpu_memory_use() const;

  const shared_ptr<SyncedMemory>& synced_mem() const;
  shared_ptr<SyncedMemory>& mutable_synced_mem(bool flush = true);

  bool is_current_valid() const {
    const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
    return mem && mem->is_valid();
  }

  void convert(Type new_type);
  void Reshape(int count);

  void* mutable_memory(Type type, bool is_gpu, bool zero_new_mem = true) {
    convert(type);
    shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
    return is_gpu ? mem->mutable_gpu_data(zero_new_mem) : mem->mutable_cpu_data(zero_new_mem);
  }

  void* current_mutable_memory(bool is_gpu, bool zero_new_mem = true) {
    shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
    return is_gpu ? mem->mutable_gpu_data(zero_new_mem) : mem->mutable_cpu_data();
  }

  const void* current_memory(bool is_gpu) {
    const shared_ptr<SyncedMemory>& mem = synced_mem();
    return is_gpu ? mem->gpu_data() : mem->cpu_data();
  }

  float asum() const;
  float sumsq() const;
  void invalidate_others();

  bool is_empty() const {
    const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
    return !mem || mem->head() == SyncedMemory::UNINITIALIZED;
  }

 public:
  static void copy_helper(bool use_gpu, int count, const void* p_src, Type src_type,
      void* p_dst, Type dst_type);  // NOLINT(runtime/references)

  std::string to_string(int indent) const;

#ifndef CPU_ONLY
  static void
  gpu_scal(int count, Type dtype, void* data, float scal, cublasHandle_t cublas_handle, bool sync);
#endif
  static void cpu_scal(int count, Type dtype, void* data, float scal);

 private:
  // numerical type stored here at a moment (might change due to conversion)
  Type type_;
  bool locked_;
  mutable bool async_state_;
  // array of projections to different types (including current type_)
  shared_ptr<vector<shared_ptr<SyncedMemory>>> synced_arrays_;
  // number of entries - comes from Blob via Reshape
  int count_;

  DISABLE_COPY_MOVE_AND_ASSIGN(Tensor);
};  // class Tensor

}  // namespace caffe

#endif /* INCLUDE_CAFFE_TENSOR_HPP_ */
