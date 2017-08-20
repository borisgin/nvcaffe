#ifdef USE_CUDNN
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
void CuDNNConvolutionLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* weight = this->blobs_[0]->template gpu_data<Ftype>();
  GPUMemory::Workspace& ws = workspace(Caffe::current_device());
  for (int i = 0; i < bottom.size(); ++i) {
    const Ftype* bottom_data = bottom[i]->gpu_data<Ftype>();
    Ftype* top_data = top[i]->mutable_gpu_data<Ftype>();
    // Filters.
    CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(),
        cudnn::dataType<Ftype>::one, fwd_bottom_descs_[i], bottom_data,
        fwd_filter_desc_, weight,
        fwd_conv_descs_[i], fwd_algo_[i], ws.data(), ws.size(),
        cudnn::dataType<Ftype>::zero, fwd_top_descs_[i], top_data));
    if (this->bias_term_) {
      const Ftype* bias_data = this->blobs_[1]->template gpu_data<Ftype>();
      CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(),
          cudnn::dataType<Ftype>::one,
          fwd_bias_desc_, bias_data,
          cudnn::dataType<Ftype>::one,
          fwd_top_descs_[i], top_data));
    }
    CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
  }  // end of for i
  const Solver* psolver = this->parent_solver();
  if (psolver == nullptr || psolver->iterations_sized() > 0) {
    // Possibly use faster algorithms by allowing larger workspace.
    use_modest_workspace_ = false;
  } else {
    Net* pnet = this->parent_net();
    if (pnet == nullptr || pnet->infer_count() > 0) {
      // Same as above in test flow
      use_modest_workspace_ = false;
    }
  }
}

template <typename Ftype, typename Btype>
void CuDNNConvolutionLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  GPUMemory::Workspace& ws = workspace(Caffe::current_device());

  // compute dE/dB = sum_c(dE/dy)
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Btype* bias_diff = this->blobs_[1]->template mutable_gpu_diff<Btype>();
    for (int i = 0; i < top.size(); ++i) {
      Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
      CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(),
          cudnn::dataType<Btype>::one, bwd_top_descs_[i], top_diff,
          cudnn::dataType<Btype>::one, bwd_bias_desc_, bias_diff));
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    }  // end of i
  }  // end of dB

  // compute dE/dW = dY * X
  if (this->param_propagate_down_[0]) {
    Btype* weight_diff = this->blobs_[0]->template mutable_gpu_diff<Btype>();
    for (int i = 0; i < top.size(); ++i) {
      Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
      const Btype* bottom_data = bottom[i]->gpu_data<Btype>();
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(),
          cudnn::dataType<Btype>::one, bwd_bottom_descs_[i], bottom_data,
          bwd_top_descs_[i], top_diff,
          bwd_conv_filter_descs_[i], bwd_filter_algo_[i], ws.data(), ws.size(),
          cudnn::dataType<Btype>::one, bwd_filter_desc_, weight_diff));
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    }  // end of i
  }

  // Backward propagate grad wrt bottom data dE/dX= dE/dY * W
  const Btype* weight = this->blobs_[0]->template gpu_data<Btype>();
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      Btype* top_diff = top[i]->mutable_gpu_diff<Btype>();
      Btype* bottom_diff = bottom[i]->mutable_gpu_diff<Btype>();
      CUDNN_CHECK(cudnnConvolutionBackwardData(Caffe::cudnn_handle(),
          cudnn::dataType<Btype>::one, bwd_filter_desc_, weight,
          bwd_top_descs_[i], top_diff,
          bwd_conv_data_descs_[i],
          bwd_data_algo_[i], ws.data(), ws.size(),
          cudnn::dataType<Btype>::zero, bwd_bottom_descs_[i], bottom_diff));
      CUDA_CHECK(cudaStreamSynchronize(Caffe::thread_stream()));
    }  // end if propagate down
  }  // end for i
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
