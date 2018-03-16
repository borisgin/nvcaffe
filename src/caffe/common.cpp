#include <glog/logging.h>
#include <syscall.h>
#include <cmath>
#include <ctime>
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
Caffe::Brew Caffe::mode_ = Caffe::GPU;
int Caffe::solver_count_ = 1;
std::vector<int> Caffe::gpus_;
int Caffe::root_device_ = -1;
int Caffe::thread_count_ = 0;
int Caffe::restored_iter_ = -1;
std::atomic<uint64_t> Caffe::root_seed_(Caffe::SEED_NOT_SET);
// NOLINT_NEXT_LINE(runtime/int)
std::atomic<size_t> Caffe::epoch_count_(static_cast<size_t>(-1L));

std::mutex Caffe::caffe_mutex_;
std::mutex Caffe::pstream_mutex_;
std::mutex Caffe::cublas_mutex_;
std::mutex Caffe::cudnn_mutex_;
std::mutex Caffe::seed_mutex_;
std::unordered_map<std::uint64_t, std::shared_ptr<Caffe>> Caffe::thread_instance_map_;


std::uint32_t lwp_id() {
#if defined(APPLE)
  return static_cast<std::uint32_t>(std::this_thread::get_id());
#else
  return static_cast<std::uint32_t>(syscall(SYS_gettid));
#endif
}

std::uint64_t lwp_dev_id() {
  std::uint64_t dev = static_cast<std::uint64_t>(Caffe::current_device());
  return lwp_id() + (dev << 32);
}

Caffe& Caffe::Get() {
  // Make sure each thread can have different values.
  // We also need to care about device id.
  std::uint64_t utid = lwp_dev_id();
  std::lock_guard<std::mutex> lock(caffe_mutex_);
  auto it = thread_instance_map_.find(utid);
  if (it != thread_instance_map_.end()) {
    return *it->second.get();
  }
  auto emp_pair = thread_instance_map_.emplace(utid, std::shared_ptr<Caffe>(new Caffe()));
  ++thread_count_;
  DLOG(INFO) << "[" << current_device()
             << "] New Caffe instance " << emp_pair.first->second.get()
             << ", count " << thread_count_ << ", thread " << lwp_id() << ", tid " << utid;
  return *emp_pair.first->second.get();
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

void Caffe::set_random_seed_int(uint64_t random_seed) {
  if (root_seed_.load() == Caffe::SEED_NOT_SET) {
    root_seed_.store(random_seed);
  } else if (random_seed == Caffe::SEED_NOT_SET) {
    return;  // i.e. root solver was previously set to 0+ and there is no need to re-generate
  }
  if (mode_ == GPU && device_count() > 0) {
    // Curand seed
    std::lock_guard<std::mutex> lock(seed_mutex_);
    if (random_seed == Caffe::SEED_NOT_SET) {
      random_seed = cluster_seedgen();
    }
    init();
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, random_seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator_, 0));
  }
  // RNG seed
  random_generator_.reset(new RNG(random_seed));
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

int Caffe::device_count() {
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}

Caffe::Caffe()
    : curand_generator_(nullptr),
      random_generator_(),
      root_solver_(true),
      device_(current_device()) {
  init();
}

void Caffe::init() {
  if (mode_ == GPU && curand_generator_ == nullptr) {
    curand_stream_ = CudaStream::create();
    CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()));
    CURAND_CHECK(curandSetStream(curand_generator_, curand_stream_->get()));
  }
}

Caffe::~Caffe() {
  int current_device;  // Just to check CUDA status:
  cudaError_t status = cudaGetDevice(&current_device);
  // Preventing crash while Caffe shutting down.
  if (status != cudaErrorCudartUnloading && curand_generator_ != nullptr) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}

size_t Caffe::min_avail_device_memory() {
  std::lock_guard<std::mutex> lock(caffe_mutex_);
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
      << stream_ << ", device " << Caffe::current_device() << ", thread "
      << lwp_id();
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
  if (group >= streams_.size()) {
    streams_.resize(group + 1UL);
  }
  if (!streams_[group]) {
    streams_[group] = CudaStream::create();
  }
  return streams_[group];
}

shared_ptr<CuBLASHandle> Caffe::th_cublas_handle(int group) {
  CHECK_GE(group, 0);
  if (group < cublas_handles_.size() && cublas_handles_[group]) {
    return cublas_handles_[group];
  }
  std::lock_guard<std::mutex> lock(cublas_mutex_);
  if (group >= cublas_handles_.size()) {
    cublas_handles_.resize(group + 1UL);
  }
  if (!cublas_handles_[group]) {
    cublas_handles_[group] = make_shared<CuBLASHandle>(pstream(group)->get());
  }
  return cublas_handles_[group];
}

#ifdef USE_CUDNN
cudnnHandle_t Caffe::th_cudnn_handle(int group) {
  CHECK_GE(group, 0);
  if (group < cudnn_handles_.size() && cudnn_handles_[group]) {
    return cudnn_handles_[group]->get();
  }
  std::lock_guard<std::mutex> lock(cudnn_mutex_);
  if (group >= cudnn_handles_.size()) {
    cudnn_handles_.resize(group + 1UL);
  }
  if (!cudnn_handles_[group]) {
    cudnn_handles_[group] = make_shared<CuDNNHandle>(pstream(group)->get());
  }
  return cudnn_handles_[group]->get();
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

const double  TypedConsts<double>::zero = 0.0;
const double  TypedConsts<double>::one = 1.0;
const float   TypedConsts<float>::zero = 0.0f;
const float   TypedConsts<float>::one = 1.0f;
const float16 TypedConsts<float16>::zero = 0.0f;
const float16 TypedConsts<float16>::one = 1.0f;
const int     TypedConsts<int>::zero = 0;
const int     TypedConsts<int>::one = 1;

CuBLASHandle::CuBLASHandle() : handle_(nullptr) {
  if (Caffe::device_count() > 0) {
    CUBLAS_CHECK(cublasCreate(&handle_));
  }
}
CuBLASHandle::CuBLASHandle(cudaStream_t stream) : handle_(nullptr) {
  if (Caffe::device_count() > 0) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
  }
}
CuBLASHandle::~CuBLASHandle() {
  if (Caffe::device_count() > 0) {
    CUBLAS_CHECK(cublasDestroy(handle_));
  }
}
#ifdef USE_CUDNN
CuDNNHandle::CuDNNHandle(cudaStream_t stream) : handle_(nullptr) {
  if (Caffe::device_count() > 0) {
    CUDNN_CHECK(cudnnCreate(&handle_));
    CUDNN_CHECK(cudnnSetStream(handle_, stream));
  }
}
CuDNNHandle::~CuDNNHandle() {
  if (Caffe::device_count() > 0) {
    CUDNN_CHECK(cudnnDestroy(handle_));
  }
}
#endif

Caffe::Properties& Caffe::props() {
  static Caffe::Properties props_;
  return props_;
}

Caffe::Properties::Properties() :
      init_time_(std::time(nullptr)),
      main_thread_id_(lwp_id()),
      caffe_version_(AS_STRING(CAFFE_VERSION)) {
  const std::vector<int>& gpus = Caffe::gpus();
  const int count = gpus.size();
  if (count == 0) {
    return;
  }
  compute_capabilities_.resize(count);
  cudaDeviceProp device_prop;
  for (int gpu = 0; gpu < compute_capabilities_.size(); ++gpu) {
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, gpus[gpu]));
    compute_capabilities_[gpu] = device_prop.major * 100 + device_prop.minor;
    DLOG(INFO) << "GPU " << gpus[gpu] << " '" << device_prop.name
               << "' has compute capability " << device_prop.major << "." << device_prop.minor;
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

#ifndef NO_NVML
namespace nvml {

std::mutex NVMLInit::m_;

NVMLInit::NVMLInit() {
  if (nvmlInit() != NVML_SUCCESS) {
    LOG(ERROR) << "NVML failed to initialize";
  } else {
    LOG(INFO) << "NVML initialized, thread " << lwp_id();
  }
}

NVMLInit::~NVMLInit() {
  nvmlShutdown();
}

// set the CPU affinity for this thread
void setCpuAffinity(int device) {
  std::lock_guard<std::mutex> lock(NVMLInit::m_);
  static NVMLInit nvml_init_;

  char pciBusId[16];
  CUDA_CHECK(cudaDeviceGetPCIBusId(pciBusId, 16, device));
  nvmlDevice_t nvml_device;

  if (nvmlDeviceGetHandleByPciBusId(pciBusId, &nvml_device) != NVML_SUCCESS ||
      nvmlDeviceSetCpuAffinity(nvml_device) != NVML_SUCCESS) {
    LOG(ERROR) << "NVML failed to set CPU affinity on device " << device
               << ", thread " << lwp_id();
  } else {
    LOG(INFO) << "NVML succeeded to set CPU affinity on device " << device
               << ", thread " << lwp_id();
  }
}

}  // namespace nvml
#endif  // NO_NVML

}  // namespace caffe
