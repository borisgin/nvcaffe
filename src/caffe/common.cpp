#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <ios>
#include <memory>

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/rng.hpp"
#if defined(USE_CUDNN)
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {

// Must be set before brewing
int Caffe::root_device_ = -1;
int Caffe::thread_count_ = 0;
int Caffe::restored_iter_ = -1;
std::atomic<uint64_t> Caffe::root_seed_(Caffe::SEED_NOT_SET);

std::mutex Caffe::caffe_mutex_;
std::mutex Caffe::pstream_mutex_;
std::mutex Caffe::cublas_mutex_;
std::mutex Caffe::cudnn_mutex_;
std::mutex Caffe::seed_mutex_;

Caffe& Caffe::Get() {
  // Make sure each thread can have different values.
  static thread_local unique_ptr<Caffe> thread_instance_;
  if (!thread_instance_) {
    std::lock_guard<std::mutex> lock(caffe_mutex_);
    if (!thread_instance_) {
      thread_instance_.reset(new Caffe());
      ++thread_count_;
    }
  }
  return *(thread_instance_.get());
}

// random seeding
uint64_t cluster_seedgen(void) {
  uint64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = static_cast<uint64_t>(getpid());
  s = static_cast<uint64_t>(time(NULL));
  seed = static_cast<uint64_t>(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

void Caffe::set_root_seed(uint64_t random_seed) {
  if (random_seed != Caffe::SEED_NOT_SET) {
    root_seed_.store(random_seed);
    set_random_seed(random_seed);
  }
}

void Caffe::set_random_seed(uint64_t random_seed) {
  if (root_seed_.load() == Caffe::SEED_NOT_SET) {
    root_seed_.store(random_seed);
  } else if (random_seed == Caffe::SEED_NOT_SET) {
    return;  // i.e. root solver was previously set to 0+ and there is no need to re-generate
  }
#ifndef CPU_ONLY
  {
    // Curand seed
    std::lock_guard<std::mutex> lock(seed_mutex_);
    if (random_seed == Caffe::SEED_NOT_SET) {
      random_seed = cluster_seedgen();
    }
    curandGenerator_t curand_generator_handle = curand_generator();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_handle, random_seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator_handle, 0));
  }
#endif
  // RNG seed
  Get().random_generator_.reset(new RNG(random_seed));
}

uint64_t Caffe::next_seed() {
  return (*caffe_rng())();
}

void Caffe::set_restored_iter(int val) {
  std::lock_guard<std::mutex> lock(caffe_mutex_);
  restored_iter_ = val;
}

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), root_solver_(true) { }

Caffe::~Caffe() { }

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

std::string Caffe::DeviceQuery() {
  NO_GPU;
  return std::string();
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(uint64_t seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU), solver_count_(1), root_solver_(true) {
  CURAND_CHECK_ARG(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT),
      current_device());
  CURAND_CHECK_ARG(curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()),
      current_device());
  curand_stream_ = pstream();
  CURAND_CHECK_ARG(curandSetStream(curand_generator_, curand_stream_->get()), current_device());
}

Caffe::~Caffe() {
  int current_device;  // Just to check CUDA status:
  cudaError_t status = cudaGetDevice(&current_device);
  // Preventing crash while Caffe shutting down.
  if (status != cudaErrorCudartUnloading) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}

size_t Caffe::min_avail_device_memory() {
  size_t ret = 0UL;
  const std::vector<int>& cur_gpus = gpus();
  int cur_device;
  size_t gpu_bytes, total_memory;
  CUDA_CHECK(cudaGetDevice(&cur_device));
  GPUMemory::GetInfo(&ret, &total_memory, true);
  for (int gpu : cur_gpus) {
    if (gpu != cur_device) {
      CUDA_CHECK(cudaSetDevice(gpu));
      GPUMemory::GetInfo(&gpu_bytes, &total_memory, true);
      if (gpu_bytes < ret) {
        ret = gpu_bytes;
      }
    }
  }
  CUDA_CHECK(cudaSetDevice(cur_device));
  return ret;
}

CudaStream::CudaStream(bool high_priority) {
  if (high_priority) {
    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, greatestPriority));
  } else {
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }
  DLOG(INFO) << "New " << (high_priority ? "high priority " : "") << "stream "
      << stream_ << ", device " << current_device() << ", thread " << std::this_thread::get_id();
}

CudaStream::~CudaStream() {
  int current_device;  // Just to check CUDA status:
  cudaError_t status = cudaGetDevice(&current_device);
  // Preventing dead lock while Caffe shutting down.
  if (status != cudaErrorCudartUnloading) {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }
}

shared_ptr<CudaStream> Caffe::pstream(int group) {
  CHECK_GE(group, 0);
  if (group < streams_.size() && streams_[group]) {
    return streams_[group];
  }
  std::lock_guard<std::mutex> lock(pstream_mutex_);
  streams_.resize(group + 1UL);
  streams_[group] = CudaStream::create();
  return streams_[group];
}

shared_ptr<CudaStream> Caffe::pstream_aux(int id) {
  CHECK_GE(id, 0);
  if (id < streams_aux_.size() && streams_aux_[id]) {
    return streams_aux_[id];
  }
  std::lock_guard<std::mutex> lock(pstream_mutex_);
  streams_aux_.resize(id + 1UL);
  streams_aux_[id] = CudaStream::create();
  return streams_aux_[id];
}

shared_ptr<CuBLASHandle> Caffe::th_cublas_handle(int group) {
  CHECK_GE(group, 0);
  if (group < cublas_handles_.size() && cublas_handles_[group]) {
    return cublas_handles_[group];
  }
  std::lock_guard<std::mutex> lock(cublas_mutex_);
  cublas_handles_.resize(group + 1UL);
  shared_ptr<CuBLASHandle>& cublas_handle = cublas_handles_[group];
  if (!cublas_handle) {
    cublas_handle = make_shared<CuBLASHandle>(pstream(group)->get());
  }
  return cublas_handle;
}

#ifdef USE_CUDNN
cudnnHandle_t Caffe::th_cudnn_handle(int group) {
  CHECK_GE(group, 0);
  if (group < cudnn_handles_.size() && cudnn_handles_[group]) {
    return cudnn_handles_[group]->get();
  }
  std::lock_guard<std::mutex> lock(cudnn_mutex_);
  cudnn_handles_.resize(group + 1UL);
  shared_ptr<CuDNNHandle>& cudnn_handle = cudnn_handles_[group];
  if (!cudnn_handle) {
    cudnn_handle = make_shared<CuDNNHandle>(pstream(group)->get());
  }
  return cudnn_handle->get();
}
#endif

void Caffe::SetDevice(const int device_id) {
  root_device_ = device_id;
  CUDA_CHECK(cudaSetDevice(root_device_));
}

std::string Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  std::ostringstream os;
  if (cudaSuccess != cudaGetDevice(&device)) {
    os << "No cuda device present." << std::endl;
  } else {
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    os << "Device id:                     " << device << std::endl;
    os << "Major revision number:         " << prop.major << std::endl;
    os << "Minor revision number:         " << prop.minor << std::endl;
    os << "Name:                          " << prop.name << std::endl;
    os << "Total global memory:           " << prop.totalGlobalMem << std::endl;
    os << "Total shared memory per block: " << prop.sharedMemPerBlock << std::endl;
    os << "Total registers per block:     " << prop.regsPerBlock << std::endl;
    os << "Warp size:                     " << prop.warpSize << std::endl;
    os << "Maximum memory pitch:          " << prop.memPitch << std::endl;
    os << "Maximum threads per block:     " << prop.maxThreadsPerBlock << std::endl;
    os << "Maximum dimension of block:    "
        << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
        << prop.maxThreadsDim[2] << std::endl;
    os << "Maximum dimension of grid:     "
        << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
        << prop.maxGridSize[2] << std::endl;
    os << "Clock rate:                    " << prop.clockRate << std::endl;
    os << "Total constant memory:         " << prop.totalConstMem << std::endl;
    os << "Texture alignment:             " << prop.textureAlignment << std::endl;
    os << "Concurrent copy and execution: "
        << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
    os << "Number of multiprocessors:     " << prop.multiProcessorCount << std::endl;
    os << "Kernel execution timeout:      "
        << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
  }
  return os.str();
}

bool Caffe::CheckDevice(const int device_id) {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
            (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
  return r;
}

int Caffe::FindDevice(const int start_id) {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(uint64_t seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG()
    : generator_(new Generator()) {}

Caffe::RNG::RNG(uint64_t seed)
    : generator_(new Generator(seed)) {}

Caffe::RNG::RNG(const RNG& other)
    : generator_(other.generator_) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#endif  // CPU_ONLY

const double TypedConsts<double>::zero = 0.0;
const double TypedConsts<double>::one = 1.0;

const float TypedConsts<float>::zero = 0.0f;
const float TypedConsts<float>::one = 1.0f;

#ifndef CPU_ONLY
const float16 TypedConsts<float16>::zero = 0.0f;
const float16 TypedConsts<float16>::one = 1.0f;
#endif

const int TypedConsts<int>::zero = 0;
const int TypedConsts<int>::one = 1;

#ifndef CPU_ONLY
CuBLASHandle::CuBLASHandle() {
  CUBLAS_CHECK(cublasCreate(&handle_));
}
CuBLASHandle::CuBLASHandle(cudaStream_t stream) {
  CUBLAS_CHECK(cublasCreate(&handle_));
  CUBLAS_CHECK(cublasSetStream(handle_, stream));
}
CuBLASHandle::~CuBLASHandle() {
  CUBLAS_CHECK(cublasDestroy(handle_));
}
#ifdef USE_CUDNN
CuDNNHandle::CuDNNHandle(cudaStream_t stream) {
  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnSetStream(handle_, stream));
}
CuDNNHandle::~CuDNNHandle() {
  CUDNN_CHECK(cudnnDestroy(handle_));
}
#endif
#endif

Caffe::Properties Caffe::props_;

Caffe::Properties::Properties() :
      init_time_(std::time(nullptr)),
      main_thread_id_(std::this_thread::get_id()),
      caffe_version_(AS_STRING(CAFFE_VERSION)) {
#ifndef CPU_ONLY
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  compute_capabilities_.resize(count);
  cudaDeviceProp device_prop;
  for (int gpu = 0; gpu < compute_capabilities_.size(); ++gpu) {
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, gpu));
    compute_capabilities_[gpu] = device_prop.major * 100 + device_prop.minor;
    DLOG(INFO) << "GPU " << gpu << " '" << device_prop.name << "' has compute capability "
        << device_prop.major << "." << device_prop.minor;
  }
#ifdef USE_CUDNN
  cudnn_version_ = std::to_string(cudnnGetVersion());
#else
  cudnn_version_ = "USE_CUDNN is not defined";
#endif
  shared_ptr<CuBLASHandle> phandle = Caffe::short_term_cublas_phandle();
  int cublas_version = 0;
  CUBLAS_CHECK(cublasGetVersion(phandle->get(), &cublas_version));
  cublas_version_ = std::to_string(cublas_version);

  int cuda_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_version));
  cuda_version_ = std::to_string(cuda_version);

  int cuda_driver_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
  cuda_driver_version_ = std::to_string(cuda_driver_version);
#endif
}

std::string Caffe::time_from_init() {
  std::ostringstream os;
  os.unsetf(std::ios_base::floatfield);
  os.precision(4);
  double span = std::difftime(std::time(NULL), init_time());
  const double mn = 60.;
  const double hr = 3600.;
  if (span < mn) {
    os << span << "s";
  } else if (span < hr) {
    int m = static_cast<int>(span / mn);
    double s = span - m * mn;
    os << m << "m " << s << "s";
  } else {
    int h = static_cast<int>(span / hr);
    int m = static_cast<int>((span - h * hr) / mn);
    double s = span - h * hr - m * mn;
    os << h << "h " << m << "m " << s << "s";
  }
  return os.str();
}

#ifndef CPU_ONLY
#ifndef NO_NVML
namespace nvml {

std::mutex NVMLInit::m_;

NVMLInit::NVMLInit() {
  if (nvmlInit() != NVML_SUCCESS) {
    LOG(ERROR) << "NVML failed to initialize";
    return;
  } else {
    LOG(INFO) << "NVML initialized, thread " << std::this_thread::get_id();
  }
  unsigned int deviceCount = 0U;
  if (nvmlDeviceGetCount(&deviceCount) == NVML_SUCCESS) {
    for (unsigned int id = 0; id < deviceCount; ++id) {
      if (nvmlDeviceGetHandleByIndex(id, &device_) != NVML_SUCCESS ||
          nvmlDeviceSetCpuAffinity(device_) != NVML_SUCCESS) {
          LOG(ERROR) << "NVML failed to set CPU affinity on device " << id
              << ", thread " << std::this_thread::get_id();
      }
    }
  } else {
    LOG(ERROR) << "nvmlDeviceGetCount failed, thread " << std::this_thread::get_id();
  }
}

NVMLInit::~NVMLInit() {
  nvmlShutdown();
}

// set the CPU affinity for this thread
void setCpuAffinity() {
  std::lock_guard<std::mutex> lock(NVMLInit::m_);
  static thread_local NVMLInit nvml_init_;
}

}  // namespace nvml
#endif  // NO_NVML
#endif

}  // namespace caffe
