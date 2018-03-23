#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/type.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX_ELEM_TO_SHOW 20UL

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
void SyncedMemory::MallocHost(void** ptr, size_t size, bool* use_cuda) {
  if (Caffe::mode() == Caffe::GPU) {
    shared_lock<shared_mutex> lock(GPUMemory::read_write_mutex());
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
  } else {
    *ptr = malloc(size);
    *use_cuda = false;
  }
}

void SyncedMemory::FreeHost(void* ptr, bool use_cuda) {
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
  } else {
    free(ptr);
  }
}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    shared_lock<shared_mutex> lock(GPUMemory::read_write_mutex());
    FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  if (gpu_ptr_ && own_gpu_data_) {
//#ifdef DEBUG
//    cudaPointerAttributes attr;
//    cudaError_t status = cudaPointerGetAttributes(&attr, gpu_ptr_);
//    if (status == cudaSuccess) {
//      CHECK_EQ(attr.memoryType, cudaMemoryTypeDevice);
//      CHECK_EQ(attr.device, device_);
//    }
//#endif
    GPUMemory::deallocate(gpu_ptr_, device_);
  }
}

void SyncedMemory::to_cpu(bool copy_from_gpu) {
  switch (head_) {
    case UNINITIALIZED:
      MallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      caffe_memset(size_, 0, cpu_ptr_);
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    case HEAD_AT_GPU:
      if (cpu_ptr_ == NULL) {
        MallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
        own_cpu_data_ = true;
      }
      if (copy_from_gpu) {
        caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
        head_ = SYNCED;
      } else {
        head_ = HEAD_AT_CPU;
      }
      break;
    case HEAD_AT_CPU:
    case SYNCED:
      break;
  }
}

void SyncedMemory::to_gpu(bool copy_from_cpu, int group) {
  switch (head_) {
    case UNINITIALIZED:
      CUDA_CHECK(cudaGetDevice(&device_));
      pstream_ = Caffe::thread_pstream(group);
      GPUMemory::allocate(&gpu_ptr_, size_, device_, pstream_);
      caffe_gpu_memset(size_, 0, gpu_ptr_, group);
      head_ = HEAD_AT_GPU;
      own_gpu_data_ = true;
      break;
    case HEAD_AT_CPU:
      if (gpu_ptr_ == NULL) {
        CUDA_CHECK(cudaGetDevice(&device_));
        pstream_ = Caffe::thread_pstream(group);
        GPUMemory::allocate(&gpu_ptr_, size_, device_, pstream_);
        own_gpu_data_ = true;
      }
      if (copy_from_cpu) {
        caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_, group);
        head_ = SYNCED;
      } else {
        head_ = HEAD_AT_GPU;
      }
      break;
    case HEAD_AT_GPU:
    case SYNCED:
      break;
  }
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*) cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    FreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data(int group) {
  to_gpu(true, group);
  return (const void*) gpu_ptr_;
}

void SyncedMemory::set_gpu_data(void* data) {
  CHECK(data);
  if (gpu_ptr_ && own_gpu_data_) {
    GPUMemory::deallocate(gpu_ptr_, device_);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
}

void* SyncedMemory::mutable_cpu_data(bool copy_from_gpu) {
  to_cpu(copy_from_gpu);
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data(bool copy_from_cpu, int group) {
  to_gpu(copy_from_cpu, group);
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
}

std::string SyncedMemory::to_string(int indent, Type type) {  // debug helper
  const std::string idt(indent, ' ');
  std::ostringstream os;
  os << idt << "SyncedMem " << this << ", size: " << size_ << ", type: " << Type_Name(type)
     << std::endl;
  os << idt << "head_: ";
  switch (head_) {
    case UNINITIALIZED:
      os << "UNINITIALIZED";
      break;
    case HEAD_AT_CPU:
      os << "HEAD_AT_CPU";
      break;
    case HEAD_AT_GPU:
      os << "HEAD_AT_GPU";
      break;
    case SYNCED:
      os << "SYNCED";
      break;
    default:
      os << "???";
      break;
  }
  os << std::endl;
  os << idt << "cpu_ptr_, gpu_ptr_: " << cpu_ptr_ << " " << gpu_ptr_ << std::endl;
  os << idt << "own_cpu_data_, own_gpu_data_: " << own_cpu_data_ << " " << own_gpu_data_
     << std::endl;
  os << idt << "cpu_malloc_use_cuda_, gpu_device_: " << cpu_malloc_use_cuda_ << " " << device_
     << std::endl;
  os << idt << "valid_: " << valid_ << std::endl;

  const void* data = cpu_data();
  if (is_type<float>(type)) {
    const float* fdata = static_cast<const float*>(data);
    size_t n = std::min(size_ / sizeof(float), MAX_ELEM_TO_SHOW);
    os << idt << "First " << n << " elements:";
    for (size_t i = 0; i < n; ++i) {
      os << " " << fdata[i];
    }
    os << std::endl;
    os << idt << "First corrupted elements (if any):";
    int j = 0;
    for (size_t i = 0; i < size_ / sizeof(float) && j < MAX_ELEM_TO_SHOW; ++i) {
      if (isinf(fdata[i]) || isnan(fdata[i])) {
        os << idt << i << "->" << fdata[i] << " ";
        ++j;
      }
    }
    os << std::endl;
  } else if (is_type<float16>(type)) {
    const float16* fdata = static_cast<const float16*>(data);
    size_t n = std::min(size_ / sizeof(float16), MAX_ELEM_TO_SHOW);
    os << idt << "First " << n << " elements:";
    for (size_t i = 0; i < n; ++i) {
      os << " " << float(fdata[i]);
    }
    os << std::endl;
    os << idt << "First corrupted elements (if any):";
    int j = 0;
    for (size_t i = 0; i < size_ / sizeof(float16) && j < MAX_ELEM_TO_SHOW; ++i) {
      if (isinf(fdata[i]) || isnan(fdata[i])) {
        os << i << "->" << float(fdata[i]) << " ";
        ++j;
      }
    }
    os << std::endl;
  } else if (is_type<double>(type)) {
    const double* fdata = static_cast<const double*>(data);
    size_t n = std::min(size_ / sizeof(double), MAX_ELEM_TO_SHOW);
    os << idt << "First " << n << " elements:";
    for (size_t i = 0; i < n; ++i) {
      os << " " << fdata[i];
    }
  } else if (is_type<unsigned int>(type)) {
    const unsigned int* fdata = static_cast<const unsigned int*>(data);
    size_t n = std::min(size_ / sizeof(unsigned int), MAX_ELEM_TO_SHOW);
    os << idt << "First " << n << " elements:";
    for (size_t i = 0; i < n; ++i) {
      os << " " << fdata[i];
    }
  } else if (is_type<int>(type)) {
    const int* fdata = static_cast<const int*>(data);
    size_t n = std::min(size_ / sizeof(int), MAX_ELEM_TO_SHOW);
    os << idt << "First " << n << " elements:";
    for (size_t i = 0; i < n; ++i) {
      os << " " << fdata[i];
    }
  } else {
    LOG(FATAL) << "Unsupported data type: " << Type_Name(type);
  }
  os << std::endl;
  return os.str();
}

}  // namespace caffe
