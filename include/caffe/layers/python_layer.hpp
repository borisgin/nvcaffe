#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

inline void PyErrReport(bool fatal = false) {
  PyObject *ptype, *pvalue, *ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  if (pvalue != nullptr) {
    const char* message = PyString_AsString(pvalue);
    if (fatal) {
      LOG(FATAL) << message;
    } else {
      LOG(ERROR) << message;
    }
  }
}

#define PYTHON_CALL_BEGIN                       \
try {                                           \
  std::lock_guard<std::mutex> lock(mutex());    \
  Py_BEGIN_ALLOW_THREADS;                       \
  PyGILState_STATE state = PyGILState_Ensure();

#define PYTHON_CALL_END                         \
  PyGILState_Release(state);                    \
  Py_END_ALLOW_THREADS;                         \
} catch (...) {                                 \
  PyErrReport();                                \
}

template <typename Ftype, typename Btype>
class PythonLayer : public Layer<Ftype, Btype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Ftype, Btype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) {
    std::lock_guard<std::mutex> lock(mutex());
    self_.attr("param_str") = bp::str(this->layer_param_.python_param().param_str());
    self_.attr("phase") = static_cast<int>(this->phase_);
    self_.attr("setup")(bottom, top);
  }

  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) {
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

  static std::mutex& init_mutex() {
    return im_;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    PYTHON_CALL_BEGIN
    self_.attr("forward")(bottom, top);
    PYTHON_CALL_END
  }

  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
    PYTHON_CALL_BEGIN
    self_.attr("backward")(top, propagate_down, bottom);
    PYTHON_CALL_END
  }

 private:
  bp::object self_;
  static std::mutex m_, im_;
};

template <typename Ftype, typename Btype>
std::mutex PythonLayer<Ftype, Btype>::m_;

template <typename Ftype, typename Btype>
std::mutex PythonLayer<Ftype, Btype>::im_;

}  // namespace caffe

#endif
