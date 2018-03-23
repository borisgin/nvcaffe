#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <thread>
#include <unordered_map>
#include <vector>

#include "caffe/common.hpp"

namespace cub {
  class CachingDeviceAllocator;
}

namespace caffe {

struct GPUMemory {
  static void GetInfo(size_t* free_mem, size_t* used_mem, bool with_update = false) {
    return mgr_.GetInfo(free_mem, used_mem, with_update);
  }

  static int current_device() {
    if (Caffe::device_count() == 0) {
      return INVALID_DEVICE;
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }

  template <class Any>
  static void allocate(Any** ptr, size_t size, int device, const shared_ptr<CudaStream>& pstream) {
    if (!try_allocate(reinterpret_cast<void**>(ptr), size, device, pstream)) {
      LOG(FATAL) << "Failed to allocate " << size << " bytes on device " << device
          << ". " << mgr_.report_dev_info(device);
    }
  }

  static void deallocate(void* ptr, int device = current_device()) {
    mgr_.deallocate(ptr, device);
  }

  static bool try_allocate(void** ptr, size_t size, int device,
                           const shared_ptr<CudaStream>& pstream) {
    return mgr_.try_allocate(ptr, size, device, pstream);
  }

  static shared_mutex& read_write_mutex() {
    return mutex_;
  }

  // Scope initializes global Memory Manager for a given scope.
  // It's instantiated in test(), train() and time() Caffe brewing functions
  // as well as in unit tests main().
  struct Scope {
    Scope(const std::vector<int>& gpus, bool debug = false) {
      mgr_.init(gpus, debug);
    }
    ~Scope() {
    }
  };

  struct Workspace {
    Workspace()
      : ptr_(nullptr), size_(0), device_(-1) {}

    Workspace(size_t size, int device)
      : ptr_(nullptr), size_(size), device_(device) {
      reserve(size_, device);
    }

    ~Workspace() {
      if (ptr_ != nullptr) {
        mgr_.deallocate(ptr_, device_);
      }
    }

    void* data() const {
      return ptr_;
    }

    size_t size() const { return size_; }
    int device() const { return device_; }
    bool empty() const { return ptr_ == nullptr; }
    bool safe_reserve(size_t size, int device = current_device());
    bool try_reserve(size_t size, int device = current_device());

    void reserve(size_t size, int device = current_device()) {
      if (!try_reserve(size, device)) {
        LOG(FATAL) << "Out of memory: failed to allocate " << size
            << " bytes on device " << device;
      }
    }

    void release() {
      if (ptr_ != nullptr) {
        mgr_.deallocate(ptr_, device_);
        ptr_ = nullptr;
        size_ = 0;
      }
    }

    void zero_mem() {
      cudaStream_t stream = Caffe::thread_stream();
      CUDA_CHECK(cudaMemsetAsync(ptr_, 0, size_, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

   private:
    void* ptr_;
    size_t size_;
    int device_;
    shared_ptr<CudaStream> pstream_;

    DISABLE_COPY_MOVE_AND_ASSIGN(Workspace);
  };

  struct PinnedBuffer {
    explicit PinnedBuffer(size_t size);
    ~PinnedBuffer();

    void* get() {
      return dptr_;
    }

   private:
    void* hptr_;
    void* dptr_;

    DISABLE_COPY_MOVE_AND_ASSIGN(PinnedBuffer);
  };

 private:
  struct Manager {
    Manager();
    ~Manager();
    void lazy_init(int device);
    void GetInfo(size_t* free_mem, size_t* used_mem, bool with_update);
    void deallocate(void* ptr, int device);
    bool try_allocate(void** ptr, size_t size, int device, const shared_ptr<CudaStream>& pstream);
    void init(const std::vector<int>&, bool);
    void reset();
    std::string report_dev_info(int device);

    bool debug_;

   private:
    struct DevInfo {
      DevInfo() {
        free_ = total_ = flush_count_ = 0;
      }
      size_t free_;
      size_t total_;
      unsigned flush_count_;
    };

    void update_dev_info(int device);

    vector<DevInfo> dev_info_;
    bool initialized_;
    std::unique_ptr<cub::CachingDeviceAllocator> cub_allocator_;
    vector<size_t> update_thresholds_;

    static const unsigned int BIN_GROWTH;  ///< Geometric growth factor
    static const unsigned int MIN_BIN;  ///< Minimum bin
    static const unsigned int MAX_BIN;  ///< Maximum bin
    static const size_t MAX_CACHED_BYTES;  ///< Maximum aggregate cached bytes
    static const size_t MAX_CACHED_SIZE;  ///< 2^MAX_BIN
  };

  static shared_mutex mutex_;
  static mutex ws_mutex_init_;
  static Manager mgr_;
  static const int INVALID_DEVICE;  ///< Default is invalid: CUB takes care

 public:
  // Workspace used by all Convolution layers one after another.
  // We carry it global to prevent unnecessary allocations/deallocations
  // because they hurt performance. It's also shared between TRAIN and TESTS nets.
  static vector<shared_ptr<Workspace>> workspace_;
  // This one is for TRAIN only:
  static vector<shared_ptr<Workspace>> weights_workspace_;

  static void Init();

  static const int WS_INITIAL_SIZE;
};

}  // namespace caffe

#endif
