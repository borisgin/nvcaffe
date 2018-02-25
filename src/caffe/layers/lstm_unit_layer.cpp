#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template<typename Ftype, typename Btype>
void LSTMUnitLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int num_instances = bottom[0]->shape(1);
  for (int i = 0; i < bottom.size(); ++i) {
    if (i == 2) {
      CHECK_EQ(2, bottom[i]->num_axes());
    } else {
      CHECK_EQ(3, bottom[i]->num_axes());
    }
    CHECK_EQ(1, bottom[i]->shape(0));
    CHECK_EQ(num_instances, bottom[i]->shape(1));
  }
  hidden_dim_ = bottom[0]->shape(2);
  CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
  X_acts_->ReshapeLike(*bottom[1]);
}

template<typename Ftype, typename Btype>
void LSTMUnitLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Ftype* C_prev = bottom[0]->cpu_data<Ftype>();
  const Ftype* X = bottom[1]->cpu_data<Ftype>();
  const Ftype* cont = bottom[2]->cpu_data<Ftype>();
  Ftype* C = top[0]->mutable_cpu_data<Ftype>();
  Ftype* H = top[1]->mutable_cpu_data<Ftype>();
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Ftype i = sigmoid(X[d]);
      const Ftype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Ftype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Ftype g = tanh(X[3 * hidden_dim_ + d]);
      const Ftype c_prev = C_prev[d];
      const Ftype c = f * c_prev + i * g;
      C[d] = c;
      const Ftype tanh_c = tanh(c);
      H[d] = o * tanh_c;
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    ++cont;
  }
}

template<typename Ftype, typename Btype>
void LSTMUnitLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }

  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Btype* C_prev = bottom[0]->cpu_data<Btype>();
  const Btype* X = bottom[1]->cpu_data<Btype>();
  const Btype* cont = bottom[2]->cpu_data<Btype>();
  const Btype* C = top[0]->cpu_data<Btype>();
  const Btype* H = top[1]->cpu_data<Btype>();
  const Btype* C_diff = top[0]->cpu_diff<Btype>();
  const Btype* H_diff = top[1]->cpu_diff<Btype>();
  Btype* C_prev_diff = bottom[0]->mutable_cpu_diff<Btype>();
  Btype* X_diff = bottom[1]->mutable_cpu_diff<Btype>();
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Btype i = sigmoid(X[d]);
      const Btype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Btype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Btype g = tanh(X[3 * hidden_dim_ + d]);
      const Btype c_prev = C_prev[d];
      const Btype c = C[d];
      const Btype tanh_c = tanh(c);
      Btype* c_prev_diff = C_prev_diff + d;
      Btype* i_diff = X_diff + d;
      Btype* f_diff = X_diff + 1 * hidden_dim_ + d;
      Btype* o_diff = X_diff + 2 * hidden_dim_ + d;
      Btype* g_diff = X_diff + 3 * hidden_dim_ + d;
      const Btype c_term_diff =
          C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
      *c_prev_diff = c_term_diff * f;
      *i_diff = c_term_diff * g * i * (1 - i);
      *f_diff = c_term_diff * c_prev * f * (1 - f);
      *o_diff = H_diff[d] * tanh_c * o * (1 - o);
      *g_diff = c_term_diff * i * (1 - g * g);
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    C_diff += hidden_dim_;
    H_diff += hidden_dim_;
    X_diff += x_dim;
    C_prev_diff += hidden_dim_;
    ++cont;
  }
}

INSTANTIATE_CLASS_FB(LSTMUnitLayer);
REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe
