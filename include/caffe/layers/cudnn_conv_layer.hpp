#ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
#define CAFFE_CUDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

#ifdef USE_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 2)
  #define CUDNN_GROUPING
#endif
#if CUDNN_VERSION_MIN(7, 0, 3)
  #define CUDNN_GROUPING2
#endif

/*
 * @brief cuDNN implementation of ConvolutionLayer.
 *        Fallback to ConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates convolution through forward kernels for filtering and bias
 * plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
 *
 * The CUDNN engine does not have memory overhead for matrix buffers. For many
 * input and filter regimes the CUDNN engine is faster than the CAFFE engine,
 * but for fully-convolutional models and large inputs the CAFFE engine can be
 * faster as long as it fits in memory.
*/
template<typename Ftype, typename Btype>
class CuDNNConvolutionLayer : public ConvolutionLayer<Ftype, Btype> {
  // Using all of memory may result in failure of workspace reserve.
  // NOLINT_NEXT_LINE(build/storage_class)
  static constexpr size_t PAGE_SIZE = 32 * 1024 * 1024;
  static constexpr int MAX_PARALLEL_GROUPS = Caffe::MAX_CONV_GROUPS;
  static constexpr int REQUEST_ALGO_COUNT = 1;
  static constexpr int ATTEMPTS_TO_RESERVE_WS = 3;
  static constexpr int REALLOC_COUNT = 3;

  static std::atomic<size_t> train_mem_req_all_grps_;
  static std::atomic<size_t> test_mem_req_all_grps_;
  static std::atomic<size_t> train_tmp_weights_mem_;

 public:
  explicit CuDNNConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Ftype, Btype>(param), handles_setup_(false),
        use_algo_seeker_(true), use_reshape_(true), initialized_cached_descs_(false),
        fwd_count_(0UL), bwd_count_(0UL),
        forward_math_(tpmax<Ftype, float>()), backward_data_math_(tpmax<Btype, float>()),
        backward_filter_math_(tpmax<Btype, float>()) {
#if CUDNN_VERSION_MIN(7, 0, 0)
    cudnn_math_override_ = -1;
#endif
  }

  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual ~CuDNNConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<bool>& propagate_down,
      const vector<Blob*>& bottom);

  bool handles_setup_;

  // algorithms for forward and backwards convolutions
  vector<cudnnConvolutionFwdAlgo_t> fwd_algo_;
  vector<cudnnConvolutionBwdFilterAlgo_t> bwd_filter_algo_;
  vector<cudnnConvolutionBwdDataAlgo_t> bwd_data_algo_;

#if CUDNN_VERSION_MIN(7, 0, 0)
  int cudnn_math_override_;
  vector<cudnnMathType_t> fwd_cudnn_math_, bwd_filter_cudnn_math_, bwd_data_cudnn_math_;
#endif

  vector<cudnnTensorDescriptor_t> fwd_bottom_descs_, fwd_top_descs_;
  vector<cudnnTensorDescriptor_t> bwd_bottom_descs_, bwd_top_descs_;
  cudnnTensorDescriptor_t fwd_bias_desc_, bwd_bias_desc_;
  cudnnFilterDescriptor_t fwd_filter_desc_, bwd_filter_desc_;
  vector<cudnnConvolutionDescriptor_t> fwd_conv_descs_;
  vector<cudnnConvolutionDescriptor_t> bwd_conv_data_descs_, bwd_conv_filter_descs_;

  int bottom_offset_, top_offset_, bias_offset_;

  vector<size_t> workspace_fwd_sizes_;
  vector<size_t> workspace_bwd_data_sizes_;
  vector<size_t> workspace_bwd_filter_sizes_;

 private:
  bool use_algo_seeker_;

  bool use_reshape_;
  bool initialized_cached_descs_;
  size_t fwd_count_, bwd_count_;

  vector<int> user_algos_override_;

  // When true, a small amount of workspace is allowed for algorithms
  bool use_modest_workspace() const {
    return fwd_count_ < 2UL;
  }
  bool ok_to_release() const {
    return bwd_count_ == REALLOC_COUNT;
  }

  void FindExConvAlgo(const vector<Blob*>& bottom, const vector<Blob*>& top);
  void GetConvAlgo(const vector<Blob*>& bottom, const vector<Blob*>& top,
      const size_t workspace_bytes, int pad_h, int pad_w, int stride_h, int stride_w);

  void AllocateFindExWorkspace();
  size_t AllocateWorkspace(size_t bottom_size);

  vector<cudnnTensorDescriptor_t> fwd_cached_bottom_descs_, bwd_cached_bottom_descs_;
  vector<cudnnConvolutionDescriptor_t> fwd_cached_conv_descs_,
      bwd_cached_conv_data_descs_, bwd_cached_conv_filter_descs_;
  bool IsBottomDescChanged(const vector<Blob*>& bottom, bool fwd_mode);
  bool IsConvDescChanged(const vector<Blob*>& bottom, bool fwd_mode);

  bool use_v7grouping() const {
#if defined(CUDNN_GROUPING2)
    return (this->channels_ == this->group_
         || this->channels_ == this->group_ * 2
         || this->channels_ == this->group_* 4)
        && (this->num_output_ == this->group_
         || this->num_output_ == this->group_ * 2
         || this->num_output_ == this->group_* 4);
#elif defined(CUDNN_GROUPING)
    return this->channels_ == this->num_output_ && this->channels_ == this->group_;
#else
    return false;
#endif
  }

  int groups() {
    return this->group_;
  }

  int ws_groups() {
    return use_v7grouping() ? 1 : std::min(this->group_, MAX_PARALLEL_GROUPS);
  }

  int idxg(int group) {
    return group % ws_groups();
  }

  Type forward_math_, backward_data_math_, backward_filter_math_;
  vector<bool> propagate_down_;
};

template<typename Ftype, typename Btype>
constexpr size_t CuDNNConvolutionLayer<Ftype, Btype>::PAGE_SIZE;
template<typename Ftype, typename Btype>
constexpr int CuDNNConvolutionLayer<Ftype, Btype>::MAX_PARALLEL_GROUPS;
template<typename Ftype, typename Btype>
constexpr int CuDNNConvolutionLayer<Ftype, Btype>::REQUEST_ALGO_COUNT;
template<typename Ftype, typename Btype>
constexpr int CuDNNConvolutionLayer<Ftype, Btype>::ATTEMPTS_TO_RESERVE_WS;
template<typename Ftype, typename Btype>
constexpr int CuDNNConvolutionLayer<Ftype, Btype>::REALLOC_COUNT;

template<typename Ftype, typename Btype>
std::atomic<size_t> CuDNNConvolutionLayer<Ftype, Btype>::train_mem_req_all_grps_;
template<typename Ftype, typename Btype>
std::atomic<size_t> CuDNNConvolutionLayer<Ftype, Btype>::test_mem_req_all_grps_;
template<typename Ftype, typename Btype>
std::atomic<size_t> CuDNNConvolutionLayer<Ftype, Btype>::train_tmp_weights_mem_;

#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
