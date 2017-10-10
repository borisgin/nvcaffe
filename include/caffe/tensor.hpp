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

  std::string to_string(int indent) const;

  static void copy_helper(bool use_gpu, int count, const void* p_src, Type src_type,
      void* p_dst, Type dst_type);  // NOLINT(runtime/references)

#ifndef CPU_ONLY
  static void gpu_scal(int count, Type dtype, void* data, float scal,
      cublasHandle_t cublas_handle);
#endif
  static void cpu_scal(int count, Type dtype, void* data, float scal);

 private:
  Type type() const {
    return type_;
  }

  size_t size_of() const {
    return tsize(type_) * count_;
  }

  void set(float value);
  void scale(float new_scale, void* handle = nullptr);
  void invalidate_others();
  void convert(Type new_type);
  void Reshape(int count);
  float asum() const;
  const shared_ptr<SyncedMemory>& synced_mem() const;
  shared_ptr<SyncedMemory>& mutable_synced_mem(bool flush = true);

  bool is_current_valid() const {
    const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
    return mem && mem->is_valid();
  }

  void* current_mutable_memory(bool is_gpu) {
    shared_ptr<SyncedMemory>& mem = mutable_synced_mem();
    return is_gpu ? mem->mutable_gpu_data() : mem->mutable_cpu_data();
  }

  const void* current_memory(bool is_gpu) {
    const shared_ptr<SyncedMemory>& mem = synced_mem();
    return is_gpu ? mem->gpu_data() : mem->cpu_data();
  }

  bool is_empty() const {
    const shared_ptr<SyncedMemory>& mem = synced_arrays_->at(type_);
    return !mem || mem->head() == SyncedMemory::UNINITIALIZED;
  }

#ifndef CPU_ONLY
  size_t gpu_memory_use(bool own_only = false) const;
#endif

  // numerical type stored here at a moment (might change due to conversion)
  Type type_;
  // array of projections to different types (including current type_)
  shared_ptr<vector<shared_ptr<SyncedMemory>>> synced_arrays_;
  // number of entries - comes from Blob via Reshape
  int count_;

  DISABLE_COPY_MOVE_AND_ASSIGN(Tensor);
};  // class Tensor

}  // namespace caffe

#endif /* INCLUDE_CAFFE_TENSOR_HPP_ */
