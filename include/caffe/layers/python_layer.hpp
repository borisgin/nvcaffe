#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

template <typename Ftype, typename Btype>
class PythonLayer : public Layer<Ftype, Btype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Ftype, Btype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("param_str") = bp::str(
        this->layer_param_.python_param().param_str());
    self_.attr("phase") = static_cast<int>(this->phase_);
    self_.attr("setup")(bottom, top);
  }
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("reshape")(bottom, top);
  }

  virtual inline bool ShareInParallel() const {
    return this->layer_param_.python_param().share_in_parallel();
  }

  virtual inline const char* type() const { return "Python"; }

  static std::mutex& mutex() {
    return m_;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("forward")(bottom, top);
  }
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("backward")(top, propagate_down, bottom);
  }

 private:
  bp::object self_;
  static std::mutex m_;
};

template <typename Ftype, typename Btype>
std::mutex PythonLayer<Ftype, Btype>::m_;

}  // namespace caffe

#endif
